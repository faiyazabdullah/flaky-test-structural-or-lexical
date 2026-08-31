#!/usr/bin/env python
"""02 -- baselines, the within- vs cross-project gap, and the transfer drop.

Replicates the generalisation gap reported by Haben et al. (MSR 2021) on this
dataset and fixes ``STRONG_WORD_BASELINE``, the number every later method must
beat.  Because the models trained here are exactly the ones the renaming
transfer measurement needs, this stage also emits ``out/03_transfer.json``:
train on ``code``, evaluate on ``code_renamed``, under the cross-project
regime.

Every vectoriser is fitted inside the fold.  ``C`` is tuned on an inner split of
the training rows.  Mined cues for ``bow_ablated`` are re-mined on each training
fold -- mining on the full set would leak.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from flaky.common import (JsonlCheckpoint, StageMarker, Timer, add_common_args, ctx_from_args,
                          file_digest, get_logger, load_json, progress, save_json, set_seed)
from flaky.cues import mine_cues
from flaky.cv import (BASELINE_METHODS, N_SPLITS, WORD_BASELINE_METHODS, build_featurizer,
                      fit_and_score, make_folds, score)

STAGE = "02_baselines"
REGIMES = ("within", "cross")
SUBSETS = ("full", "matched")


def rows_from(df: pd.DataFrame, text_col: str) -> list[dict]:
    return [
        {"text": t, "n_chars": c, "n_lines": l, "n_tokens": n}
        for t, c, l, n in zip(df[text_col], df["n_chars"], df["n_lines"], df["n_tokens"])
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--folds", type=int, default=N_SPLITS)
    ap.add_argument("--mined-top-k", type=int, default=200)
    args = ap.parse_args()

    ctx = ctx_from_args(args)
    log = get_logger(STAGE, ctx)
    set_seed(ctx.seed)

    prepped = ctx.data / "prepped.csv"
    df_all = pd.read_csv(prepped, keep_default_na=False, na_values=[])
    if "code_renamed" not in df_all.columns:
        log.error("prepped.csv has no code_renamed column -- run s03_cues_rename.py first")
        return 1
    cue_vocab = load_json(ctx.data / "cue_vocab.json")
    static_cues = set(cue_vocab["static_union"])
    log.info("%d rows, %d projects, %d flaky; %d static cues",
             len(df_all), df_all.project.nunique(), int(df_all.flaky.sum()), len(static_cues))

    marker = StageMarker(ctx, STAGE)
    sig = f"{file_digest(prepped)}:{ctx.seed}:{args.folds}:{args.mined_top_k}"
    if marker.matches(sig) and not args.force:
        log.info("already done (signature %s); use --force to redo", sig)
        return 0

    cp = JsonlCheckpoint(ctx.ckpt / "02_baselines.jsonl")
    if args.force:
        cp.reset()

    jobs = [(m, r, s) for s in SUBSETS for r in REGIMES for m in BASELINE_METHODS]
    results: dict = {}

    with cp:
        for method, regime, subset in progress(jobs, desc="baseline cells"):
            df = df_all if subset == "full" else df_all[df_all.matched].reset_index(drop=True)
            y = df["flaky"].to_numpy()
            groups = df["project"].to_numpy()
            if y.sum() < args.folds or len(set(groups)) < args.folds:
                log.warning("skipping %s/%s/%s: not enough positives or projects",
                            method, regime, subset)
                continue
            folds = make_folds(y, groups, regime, n_splits=args.folds, seed=ctx.seed)
            rows_code = rows_from(df, "code")
            rows_ren = rows_from(df, "code_renamed")

            fold_res = []
            for fi, (tr, te) in enumerate(folds):
                key = f"{method}|{regime}|{subset}|{fi}"
                if key in cp:
                    fold_res.append(cp.get(key))
                    continue

                drop = set()
                if method == "bow_ablated":
                    mined = mine_cues([rows_code[i]["text"] for i in tr], y[tr],
                                      top_k=args.mined_top_k)
                    drop = static_cues | set(mined)

                feat = build_featurizer(method, cue_drop=drop)
                feat.fit([rows_code[i] for i in tr], y[tr])
                Xtr = feat.transform([rows_code[i] for i in tr])
                Xte = feat.transform([rows_code[i] for i in te])
                Xte_ren = feat.transform([rows_ren[i] for i in te])

                r, _ = fit_and_score(Xtr, y[tr], groups[tr], Xte, y[te], regime,
                                     seed=ctx.seed, extra_eval={"renamed": Xte_ren})
                r["n_dropped_features"] = int(getattr(feat, "n_dropped", 0))
                r["n_features"] = int(Xtr.shape[1])
                r["n_train"] = int(len(tr))
                r["n_test_projects"] = int(len(set(groups[te])))

                # sanity check: a model retrained on the renamed variant should
                # land within noise of the original -- if it does not, the
                # renaming is not shape-preserving and is leaking information.
                if regime == "cross" and subset == "full":
                    feat2 = build_featurizer(method, cue_drop=drop)
                    feat2.fit([rows_ren[i] for i in tr], y[tr])
                    r2, _ = fit_and_score(feat2.transform([rows_ren[i] for i in tr]), y[tr],
                                          groups[tr], feat2.transform([rows_ren[i] for i in te]),
                                          y[te], regime, seed=ctx.seed)
                    r["retrained_on_renamed"] = {"ap": r2["ap"], "auroc": r2["auroc"]}

                cp.put(key, r)
                fold_res.append(r)

            results[f"{method}|{regime}|{subset}"] = fold_res

    # ---------------------------------------------------------------- summary
    def agg(fr, path=("ap",)):
        vals = []
        for r in fr:
            v = r
            for k in path:
                v = v[k]
            vals.append(float(v))
        a = np.asarray(vals, dtype=np.float64)
        return {"mean": float(np.nanmean(a)), "std": float(np.nanstd(a, ddof=1)) if len(a) > 1
                else 0.0, "folds": vals}

    summary: dict = {"n_folds": args.folds, "seed": ctx.seed, "smoke": ctx.smoke,
                     "n_rows": int(len(df_all)), "n_projects": int(df_all.project.nunique()),
                     "class_prior_full": float(df_all.flaky.mean()),
                     "class_prior_matched": float(df_all[df_all.matched].flaky.mean())
                     if df_all.matched.any() else None,
                     "methods": {}}

    for key, fr in results.items():
        method, regime, subset = key.split("|")
        cell = {
            "ap": agg(fr, ("ap",)),
            "auroc": agg(fr, ("auroc",)),
            "ap_on_renamed": agg(fr, ("renamed", "ap")),
            "C_per_fold": [r["C"] for r in fr],
            "n_features_per_fold": [r["n_features"] for r in fr],
            "n_dropped_features_per_fold": [r["n_dropped_features"] for r in fr],
            "prior_per_fold": [r["prior"] for r in fr],
        }
        if "retrained_on_renamed" in fr[0]:
            cell["retrained_on_renamed_ap"] = agg(fr, ("retrained_on_renamed", "ap"))
        summary["methods"].setdefault(method, {}).setdefault(subset, {})[regime] = cell

    # gap = AP_within - AP_cross, per method and subset
    for method, by_subset in summary["methods"].items():
        for subset, by_regime in by_subset.items():
            if "within" in by_regime and "cross" in by_regime:
                w = np.asarray(by_regime["within"]["ap"]["folds"])
                c = np.asarray(by_regime["cross"]["ap"]["folds"])
                by_regime["gap"] = {
                    "mean": float(w.mean() - c.mean()),
                    "within_mean": float(w.mean()),
                    "cross_mean": float(c.mean()),
                    "note": "folds are not paired across regimes; compare means only",
                }

    # transfer drop, cross-project regime
    transfer = {"regime": "cross", "subset": "full", "n_folds": args.folds, "methods": {}}
    for method in BASELINE_METHODS:
        cell = summary["methods"].get(method, {}).get("full", {}).get("cross")
        if cell is None:
            continue
        code = np.asarray(cell["ap"]["folds"])
        ren = np.asarray(cell["ap_on_renamed"]["folds"])
        d = code - ren
        entry = {
            "ap_code": cell["ap"],
            "ap_code_renamed": cell["ap_on_renamed"],
            "delta": {"mean": float(d.mean()),
                      "std": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
                      "folds": d.tolist()},
        }
        if "retrained_on_renamed_ap" in cell:
            rt = np.asarray(cell["retrained_on_renamed_ap"]["folds"])
            fold_std = float(code.std(ddof=1)) if len(code) > 1 else 0.0
            entry["retrained_on_renamed"] = {
                "ap": cell["retrained_on_renamed_ap"],
                "abs_diff_from_code_mean": float(abs(rt.mean() - code.mean())),
                "within_one_fold_std": bool(abs(rt.mean() - code.mean()) <= fold_std),
                "fold_std_of_code_ap": fold_std,
                "note": ("Sanity check only. Retraining on the renamed variant measures "
                         "nothing about lexical reliance -- a bag-of-tokens model simply "
                         "relearns the neutral names. It must land within noise of AP on "
                         "`code`; if it does not, the renaming is leaking."),
            }
        transfer["methods"][method] = entry

    # STRONG_WORD_BASELINE -- fixed here, not renegotiable later
    cands = {}
    for m in WORD_BASELINE_METHODS:
        cell = summary["methods"].get(m, {}).get("full", {}).get("cross")
        if cell:
            cands[m] = cell["ap"]["mean"]
    best = max(cands, key=cands.get)
    summary["STRONG_WORD_BASELINE"] = {
        "method": best,
        "ap_mean": cands[best],
        "ap_folds": summary["methods"][best]["full"]["cross"]["ap"]["folds"],
        "ap_std": summary["methods"][best]["full"]["cross"]["ap"]["std"],
        "candidates": cands,
        "regime": "cross",
        "subset": "full",
        "note": "Stopping rule S1 is evaluated against this value. Fixed in 02.",
    }

    # acceptance
    prior_full = summary["class_prior_full"]
    above_prior = {}
    for method, by_subset in summary["methods"].items():
        for subset, by_regime in by_subset.items():
            prior = prior_full if subset == "full" else summary["class_prior_matched"]
            for regime in ("within", "cross"):
                if regime in by_regime:
                    above_prior[f"{method}|{subset}|{regime}"] = (
                        by_regime[regime]["ap"]["mean"] > prior)
    bow_full = summary["methods"]["bow"]["full"]
    summary["acceptance"] = {
        "all_methods_above_class_prior": all(above_prior.values()),
        "methods_below_prior": [k for k, v in above_prior.items() if not v],
        "bow_gap_positive": bow_full["gap"]["mean"] > 0,
        "bow_gap": bow_full["gap"]["mean"],
        "fold_level_vectors_persisted": True,
    }

    save_json(ctx.out / "02_baselines.json", summary)
    save_json(ctx.out / "03_transfer.json", transfer)
    marker.write(sig)

    log.info("--- cross-project AP (full set) ---")
    for m in BASELINE_METHODS:
        cell = summary["methods"].get(m, {}).get("full", {})
        if "cross" in cell:
            log.info("  %-16s cross %.4f +- %.4f | within %.4f | gap %+.4f | renamed %.4f",
                     m, cell["cross"]["ap"]["mean"], cell["cross"]["ap"]["std"],
                     cell["within"]["ap"]["mean"], cell["gap"]["mean"],
                     cell["cross"]["ap_on_renamed"]["mean"])
    log.info("STRONG_WORD_BASELINE = %.4f (%s)", cands[best], best)
    for k, v in summary["acceptance"].items():
        if isinstance(v, bool):
            log.info("acceptance %-34s %s", k, "PASS" if v else "FAIL")
    ok = summary["acceptance"]["all_methods_above_class_prior"] and \
        summary["acceptance"]["bow_gap_positive"]
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
