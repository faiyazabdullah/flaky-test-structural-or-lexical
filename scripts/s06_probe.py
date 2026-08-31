#!/usr/bin/env python
"""06 -- representations and linear probes.

Minimal probing only.  Nullspace projection and attention analysis are Phase 2
and are deliberately absent from this codebase.

Two stages, each independently resumable:

1. hidden-state extraction into ``out/emb_{model}_{variant}_{pool}.npy``
   (fp16, ``[n, L+1, d]``, row order matching ``data/prepped.csv``);
2. ``LogisticRegression(class_weight='balanced')`` per layer under the
   cross-project regime, for every probe target.

The *full* layer curve is reported.  A selected best layer is never quoted on
its own -- where a maximum appears, the number of layers searched appears with
it.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from flaky.common import (JsonlCheckpoint, StageMarker, Timer, add_common_args, ctx_from_args,
                          file_digest, get_logger, gpu_report, load_json, progress, save_json,
                          set_seed)
from flaky.cv import make_folds
from flaky.embed import MODELS, emb_meta, extract
from flaky.structural import PROPERTIES

STAGE = "06_probe"
VARIANTS = ("code", "code_renamed")

# Probe targets.  ``eval_variant`` names the representation the fitted probe is
# scored on; ``flaky_transfer`` reuses the ``flaky_code`` fit, so it costs
# nothing extra.
TARGETS = ("flaky_code", "flaky_renamed", "control", *PROPERTIES)


def probe_layer(emb_code: str, emb_ren: str, layer: int, y: np.ndarray,
                y_control: np.ndarray, y_props: dict, groups: np.ndarray,
                folds: list, seed: int, grid: tuple, inner_k: int) -> dict:
    """All targets, all folds, for one layer.  Runs in a worker process."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import warnings

    # AP is undefined on a fold with no positives; those folds are skipped and
    # counted, not silently averaged in. The warning is noise once that is handled.
    warnings.filterwarnings("ignore", message="No positive class found in y_true")

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from flaky.cv import inner_folds, score

    Xc = np.asarray(np.load(emb_code, mmap_mode="r")[:, layer, :], dtype=np.float32)
    Xr = np.asarray(np.load(emb_ren, mmap_mode="r")[:, layer, :], dtype=np.float32) \
        if emb_ren else None

    def fit_one(X, yy, tr, te, X_eval_te=None):
        sc = StandardScaler().fit(X[tr])
        Xtr = sc.transform(X[tr])
        best, best_ap, table = grid[0], -np.inf, {}
        for C in grid:
            aps = []
            for itr, ite in inner_folds(yy[tr], groups[tr], "cross", n_splits=inner_k,
                                        seed=seed):
                if len(set(yy[tr][itr])) < 2 or len(set(yy[tr][ite])) < 2:
                    continue
                clf = LogisticRegression(C=C, class_weight="balanced", max_iter=1000,
                                         random_state=seed)
                clf.fit(Xtr[itr], yy[tr][itr])
                aps.append(score(yy[tr][ite], clf.decision_function(Xtr[ite]))["ap"])
            table[str(C)] = float(np.mean(aps)) if aps else float("nan")
            if aps and table[str(C)] > best_ap:
                best, best_ap = C, table[str(C)]
        clf = LogisticRegression(C=best, class_weight="balanced", max_iter=1000,
                                 random_state=seed)
        clf.fit(Xtr, yy[tr])
        out = {"C": best, **score(yy[te], clf.decision_function(sc.transform(X[te])))}
        if X_eval_te is not None:
            out["transfer"] = score(yy[te], clf.decision_function(sc.transform(X_eval_te[te])))
        return out

    res: dict = {"layer": layer, "targets": {}, "skipped_folds": {}}
    for name in TARGETS:
        per_fold = []
        yy_for = {"flaky_code": y, "flaky_renamed": y, "control": y_control}.get(
            name, y_props.get(name))
        for tr, te in folds:
            tr, te = np.asarray(tr), np.asarray(te)
            if yy_for is not None and (yy_for[te].sum() == 0 or yy_for[tr].sum() < 2):
                # AP is undefined without a positive in the test fold; record the
                # skip rather than averaging a meaningless 1.0 into the curve.
                res["skipped_folds"].setdefault(name, []).append(
                    {"n_pos_test": int(yy_for[te].sum()), "n_pos_train": int(yy_for[tr].sum())})
                continue
            if name == "flaky_code":
                per_fold.append(fit_one(Xc, y, tr, te, X_eval_te=Xr))
            elif name == "flaky_renamed":
                if Xr is None:
                    continue
                per_fold.append(fit_one(Xr, y, tr, te))
            elif name == "control":
                per_fold.append(fit_one(Xc, y_control, tr, te))
            else:
                per_fold.append(fit_one(Xc, y_props[name], tr, te))
        if per_fold:
            res["targets"][name] = per_fold
    return res


def agg(folds_list, path=("ap",)):
    vals = []
    for r in folds_list:
        v = r
        for k in path:
            v = v[k]
        vals.append(float(v))
    a = np.asarray(vals)
    return {"mean": float(np.nanmean(a)),
            "std": float(np.nanstd(a, ddof=1)) if len(a) > 1 else 0.0,
            "folds": vals}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--models", default="codebert,qwen1_5b",
                    help="comma-separated keys from flaky.embed.MODELS")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--c-grid", default="0.01,0.1,1.0")
    ap.add_argument("--inner-k", type=int, default=2)
    ap.add_argument("--extract-only", action="store_true")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="override the model's default batch size")
    args = ap.parse_args()

    ctx = ctx_from_args(args)
    log = get_logger(STAGE, ctx)
    set_seed(ctx.seed)
    grid = tuple(float(x) for x in args.c_grid.split(","))

    df = pd.read_csv(ctx.data / "prepped.csv", keep_default_na=False, na_values=[])
    struct = pd.read_csv(ctx.data / "structural.csv")
    assert list(struct["uid"]) == list(df["uid"]), \
        "structural.csv row order differs from prepped.csv"
    log.info("gpu: %s", gpu_report())

    y = df["flaky"].to_numpy()
    groups = df["project"].to_numpy()
    y_props = {p: struct[p].to_numpy() for p in PROPERTIES}
    # Control task: random labels matched to the class prior, fixed across folds.
    rng = np.random.RandomState(ctx.seed + 991)
    y_control = np.zeros(len(df), dtype=np.int64)
    y_control[rng.choice(len(df), size=int(y.sum()), replace=False)] = 1
    folds = make_folds(y, groups, "cross", n_splits=args.folds, seed=ctx.seed)

    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    results: dict = {"seed": ctx.seed, "smoke": ctx.smoke, "n_rows": int(len(df)),
                     "n_folds": args.folds, "regime": "cross",
                     "class_prior": float(y.mean()),
                     "control_prior": float(y_control.mean()),
                     "gpu": gpu_report(), "models": {}}

    # ---------------------------------------------------------- extraction
    extraction_meta = {}
    for key in model_keys:
        spec = MODELS[key]
        if args.batch_size:
            spec = spec.__class__(spec.key, spec.hf_id, spec.kind, spec.max_length,
                                  args.batch_size, spec.pools)
        for variant in VARIANTS:
            texts = df[variant].tolist()
            with Timer(log, f"extract {key}/{variant}"):
                meta = extract(spec, texts, df["id"].tolist(), ctx.out, ctx.ckpt, variant,
                               cache_dir=ctx.cache / "hf", logger=log, force=args.force)
            extraction_meta[f"{key}|{variant}"] = meta
            # Row identity, not just row count: a stale .npy left over from a
            # different prep would have the right shape and the wrong rows, and
            # nothing downstream would notice.
            ids_path = ctx.out / f"emb_{key}_{variant}_ids.json"
            if ids_path.exists() and not args.force:
                prev = load_json(ids_path).get("uids")
                assert prev is None or prev == df["uid"].tolist(), (
                    f"{ids_path} does not match data/prepped.csv -- the embeddings were "
                    f"extracted from a different corpus. Re-run with --force.")
            save_json(ids_path, {"uids": df["uid"].tolist(), "ids": df["id"].tolist(),
                                 "n": int(len(df))})
            for pool, path in meta["paths"].items():
                shape = emb_meta(Path(path))
                assert shape[0] == len(df), f"{path}: {shape[0]} rows, expected {len(df)}"
            log.info("%s/%s: truncation rate %.4f, peak VRAM %.2f GB",
                     key, variant, meta["truncation_rate"], meta["peak_vram_gb"])
            if key == "codebert" and meta["truncation_rate"] > 0.20:
                log.warning("CodeBERT truncation rate %.3f exceeds 20%% at max_length=512 -- "
                            "this must be stated explicitly in the write-up, not buried",
                            meta["truncation_rate"])

    results["extraction"] = extraction_meta
    if args.extract_only:
        save_json(ctx.out / "06_probe.json", results)
        log.info("--extract-only: stopping after extraction")
        return 0

    # -------------------------------------------------------------- probing
    from joblib import Parallel, delayed

    for key in model_keys:
        spec = MODELS[key]
        for pool in spec.pools:
            emb_code = ctx.out / f"emb_{key}_code_{pool}.npy"
            emb_ren = ctx.out / f"emb_{key}_code_renamed_{pool}.npy"
            n, n_layers, d = emb_meta(emb_code)
            log.info("probing %s/%s: %d layers, d=%d", key, pool, n_layers, d)

            cp = JsonlCheckpoint(ctx.ckpt / f"06_probe_{key}_{pool}.jsonl")
            if args.force:
                cp.reset()
            todo = [l for l in range(n_layers) if str(l) not in cp]
            if todo and len(todo) < n_layers:
                log.info("resuming: %d/%d layers already probed", n_layers - len(todo), n_layers)

            with cp, Timer(log, f"probe {key}/{pool} ({len(todo)} layers, {args.jobs} jobs)"):
                batch = 0
                for i in range(0, len(todo), args.jobs):
                    chunk = todo[i:i + args.jobs]
                    outs = Parallel(n_jobs=args.jobs, backend="loky")(
                        delayed(probe_layer)(str(emb_code), str(emb_ren), l, y, y_control,
                                             y_props, groups, folds, ctx.seed, grid,
                                             args.inner_k)
                        for l in chunk)
                    for l, o in zip(chunk, outs):
                        cp.put(str(l), o)
                    batch += len(chunk)
                    log.info("  layers %d/%d done", batch, len(todo))

            # ---- summarise
            per_layer = [cp.get(str(l)) for l in range(n_layers)]
            entry: dict = {"n_layers": n_layers, "hidden_size": d, "pool": pool,
                           "layers": list(range(n_layers)), "targets": {}}
            for name in TARGETS:
                curve, curve_tr = [], []
                for L in per_layer:
                    fl = L["targets"].get(name)
                    curve.append(agg(fl) if fl else None)
                    if name == "flaky_code" and fl and "transfer" in fl[0]:
                        curve_tr.append(agg(fl, ("transfer", "ap")))
                entry["targets"][name] = {"ap_by_layer": curve}
                if curve_tr:
                    entry["targets"][name]["ap_on_renamed_by_layer"] = curve_tr
                    entry["targets"][name]["transfer_delta_by_layer"] = [
                        {"mean": a["mean"] - b["mean"],
                         "folds": [x - z for x, z in zip(a["folds"], b["folds"])]}
                        for a, b in zip(curve, curve_tr)]

            # selectivity S = AP(real) - AP(control), per layer
            ctrl = entry["targets"]["control"]["ap_by_layer"]
            for name in TARGETS:
                if name == "control":
                    continue
                real = entry["targets"][name]["ap_by_layer"]
                entry["targets"][name]["selectivity_by_layer"] = [
                    {"mean": r["mean"] - c["mean"],
                     "folds": [a - b for a, b in zip(r["folds"], c["folds"])]}
                    if r and c else None for r, c in zip(real, ctrl)]

            # the maximum, always quoted with the number of layers searched
            for name in TARGETS:
                cur = entry["targets"][name]["ap_by_layer"]
                means = [c["mean"] if c else -np.inf for c in cur]
                bl = int(np.argmax(means))
                entry["targets"][name]["max"] = {
                    "layer": bl, "ap_mean": cur[bl]["mean"], "ap_std": cur[bl]["std"],
                    "ap_folds": cur[bl]["folds"], "n_layers_searched": n_layers,
                    "note": "a maximum over a searched curve, not a held-out estimate",
                }
            results["models"].setdefault(key, {})[pool] = entry
            save_json(ctx.out / "06_probe.json", results)
            log.info("%s/%s flaky_code AP: max %.4f at layer %d of %d, last layer %.4f",
                     key, pool, entry["targets"]["flaky_code"]["max"]["ap_mean"],
                     entry["targets"]["flaky_code"]["max"]["layer"], n_layers,
                     entry["targets"]["flaky_code"]["ap_by_layer"][-1]["mean"])

    # ----------------------------------------------------------- acceptance
    peaks = [m["peak_vram_gb"] for m in extraction_meta.values()]
    trunc = {k: m["truncation_rate"] for k, m in extraction_meta.items()}
    sel_ok = {}
    for key, pools in results["models"].items():
        for pool, entry in pools.items():
            bl = entry["targets"]["flaky_code"]["max"]["layer"]
            sel_ok[f"{key}|{pool}"] = (
                entry["targets"]["flaky_code"]["selectivity_by_layer"][bl]["mean"] > 0)
    results["acceptance"] = {
        "row_order_matches_prepped": True,
        "peak_vram_gb": max(peaks) if peaks else 0.0,
        "peak_vram_under_7_5gb": (max(peaks) if peaks else 0.0) < 7.5,
        "truncation_rates": trunc,
        # Phrased so True means "acceptable". 06 requires that a CodeBERT
        # truncation rate above 20% be stated explicitly rather than buried;
        # below 20% there is nothing to state.
        "codebert_truncation_within_20pct": not any(
            v > 0.20 for k, v in trunc.items() if k.startswith("codebert")),
        "selectivity_positive_at_claimed_layer": sel_ok,
    }
    save_json(ctx.out / "06_probe.json", results)
    StageMarker(ctx, STAGE).write(f"{file_digest(ctx.data / 'prepped.csv')}:{args.models}:{ctx.seed}")

    log.info("peak VRAM %.2f GB (acceptance < 7.5)", max(peaks) if peaks else 0.0)
    for k, v in sel_ok.items():
        log.info("acceptance selectivity_positive %-20s %s", k, "PASS" if v else "FAIL")
    return 0 if results["acceptance"]["peak_vram_under_7_5gb"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
