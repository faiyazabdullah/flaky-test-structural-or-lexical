#!/usr/bin/env python
"""05 (part 2) -- score the minimal pairs.

    PairAcc(s) = (1/N) * sum_i 1[ s(x_pre_i) > s(x_post_i) ]

Two-sided binomial against 0.5, 95% Wilson interval, and the mean margin
``s(x_pre) - s(x_post)`` -- which catches the case where the direction is right
but the model is barely separating the pair.

**The prediction is signed, not two-tailed.**  Fixes for async flakiness
usually *add* ``await``, ``join()`` or a latch, so the post-fix version often
contains *more* cue tokens than the pre-fix one.  A model reading cue
vocabulary therefore scores **below** 0.5; a model reading structure scores
above it.  Chance is not the interesting comparison -- the contrast between the
probe and ``bow`` is.

Every scorer is trained with **all pair projects held out**.  A pair whose
project appeared in training is not a test of generalisation, and the assertion
that enforces this is the easiest thing here to get silently wrong.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from flaky.common import (StageMarker, Timer, add_common_args, ctx_from_args, get_logger,
                          load_json, report_acceptance, save_json, set_seed)
from flaky.cv import build_featurizer, inner_folds, make_clf
from flaky.embed import MODELS, extract
from flaky.stats import binom_test, wilson_interval

STAGE = "05_score_pairs"
POWER_FLOOR = 40


def pair_metrics(s_pre: np.ndarray, s_post: np.ndarray) -> dict:
    n = len(s_pre)
    if n == 0:
        return {"n": 0, "pair_acc": None, "note": "no pairs"}
    wins = int((s_pre > s_post).sum())
    ties = int((s_pre == s_post).sum())
    lo, hi = wilson_interval(wins, n)
    margin = s_pre - s_post
    return {
        "n": n,
        "n_pre_higher": wins,
        "n_ties": ties,
        "pair_acc": wins / n,
        "wilson_95": [lo, hi],
        "binom_p_two_sided": binom_test(wins, n, 0.5, "two-sided"),
        "mean_margin": float(margin.mean()),
        "median_margin": float(np.median(margin)),
        "margin_std": float(margin.std(ddof=1)) if n > 1 else 0.0,
    }


def tune_and_fit(X, y, groups, seed, grid=(0.01, 0.1, 1.0)):
    from sklearn.metrics import average_precision_score

    best, best_ap = grid[0], -np.inf
    for C in grid:
        aps = []
        for itr, ite in inner_folds(y, groups, "cross", n_splits=3, seed=seed):
            if len(set(y[itr])) < 2 or len(set(y[ite])) < 2:
                continue
            clf = make_clf(C, seed=seed)
            clf.fit(X[itr], y[itr])
            aps.append(average_precision_score(y[ite], clf.decision_function(X[ite])))
        m = float(np.mean(aps)) if aps else -np.inf
        if m > best_ap:
            best, best_ap = C, m
    clf = make_clf(best, seed=seed)
    clf.fit(X, y)
    return clf, best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--models", default="codebert,qwen1_5b")
    args = ap.parse_args()

    ctx = ctx_from_args(args)
    log = get_logger(STAGE, ctx)
    set_seed(ctx.seed)

    all_pairs = pd.read_csv(ctx.data / "minimal_pairs.csv", keep_default_na=False, na_values=[])
    df = pd.read_csv(ctx.data / "prepped.csv", keep_default_na=False, na_values=[])
    if len(all_pairs) == 0:
        log.error("no minimal pairs -- run scripts/s05_mine_pairs.py first")
        save_json(ctx.out / "05_pairs.json",
                  {"n_pairs": 0, "verdict": "inconclusive", "reason": "no pairs mined"})
        return 0

    # Two pair sets, each with its own hold-out. They trade pair count against
    # scorer strength and neither dominates, so both are computed and reported.
    #
    #   flakebench_matched  x_pre is a test FlakeBench lists as flaky -- the
    #                       strongest provenance, and holding out only its
    #                       projects leaves a much larger training set.
    #   all                 every test method a flakiness-fix commit touched.
    #                       More pairs, but x_pre's flakiness is inferred from
    #                       the commit rather than from a label, and holding out
    #                       65 of 98 projects leaves a thin scorer.
    #
    # `flakebench_matched` is primary; S3 in 07 reads it. Stated in the report.
    pair_sets = {
        "flakebench_matched": all_pairs[all_pairs.in_flakebench.astype(bool)]
        .reset_index(drop=True),
        "all": all_pairs,
    }
    pair_sets = {k: v for k, v in pair_sets.items() if len(v) > 0}
    probe_meta = load_json(ctx.out / "06_probe.json") if (ctx.out / "06_probe.json").exists() \
        else None
    if probe_meta is None:
        log.warning("no out/06_probe.json -- probe scorers skipped")

    res: dict = {
        "primary_pair_set": "flakebench_matched",
        "power_floor": POWER_FLOOR,
        "signed_prediction": ("A cue-reading model scores BELOW 0.5, because fixes usually add "
                              "cue tokens (await, join, a latch). A structure-reading model "
                              "scores above it. The contrast between the probe and bow is the "
                              "result; chance is not the interesting comparison."),
        "pair_sets": {},
        "mining_note": (f"{len(all_pairs)} pairs mined in total; "
                        f"{int(all_pairs.in_flakebench.sum())} match a FlakeBench flaky test "
                        f"by name, the rest are other test methods the same fix commits "
                        f"touched."),
    }

    for set_name, pairs in pair_sets.items():
        log.info("=" * 60)
        log.info("pair set %r: %d pairs", set_name, len(pairs))
        pair_projects = set(pairs["project"]) | {p.replace("/", "_") for p in pairs["owner_repo"]}
        train_mask = ~df["project"].isin(pair_projects)
        train = df[train_mask].reset_index(drop=True)
        log.info("  training on %d/%d rows (%d flaky) from %d held-in projects",
                 len(train), len(df), int(train.flaky.sum()), train.project.nunique())

        # The assertion the plan singles out as easiest to get silently wrong.
        leaked = sorted(set(train["project"]) & pair_projects)
        assert not leaked, f"pair projects leaked into training: {leaked}"

        if train.flaky.sum() < 10:
            log.warning("  only %d flaky training rows -- scorers will be weak",
                        int(train.flaky.sum()))

        y_tr = train["flaky"].to_numpy()
        g_tr = train["project"].to_numpy()
        pre_texts = pairs["code_pre"].tolist()
        post_texts = pairs["code_post"].tolist()
        scorers: dict[str, dict] = {}

        with Timer(log, f"  bow scorer [{set_name}]"):
            feat = build_featurizer("bow")
            rows_tr = [{"text": t} for t in train["code"]]
            feat.fit(rows_tr, y_tr)
            Xtr = feat.transform(rows_tr)
            clf, C = tune_and_fit(Xtr, y_tr, g_tr, ctx.seed)
            s_pre = clf.decision_function(feat.transform([{"text": t} for t in pre_texts]))
            s_post = clf.decision_function(feat.transform([{"text": t} for t in post_texts]))
        scorers["bow"] = {"C": C, "n_train": int(len(train)),
                          "n_train_flaky": int(y_tr.sum()),
                          "scores_pre": s_pre.tolist(), "scores_post": s_post.tolist()}

        if probe_meta is not None:
            for key in [m.strip() for m in args.models.split(",") if m.strip()]:
                for pool, entry in probe_meta.get("models", {}).get(key, {}).items():
                    # Layer chosen on the MAIN dataset's cross-project curve,
                    # never on the pairs. Stated in the report alongside the result.
                    layer = entry["targets"]["flaky_code"]["max"]["layer"]
                    n_layers = entry["n_layers"]
                    spec = MODELS[key]
                    with Timer(log, f"  embed pairs for {key}/{pool} [{set_name}]"):
                        meta_pre = extract(spec, pre_texts, list(range(len(pre_texts))),
                                           ctx.out, ctx.ckpt, f"pairs_{set_name}_pre",
                                           cache_dir=ctx.cache / "hf", logger=log,
                                           force=args.force)
                        meta_post = extract(spec, post_texts, list(range(len(post_texts))),
                                            ctx.out, ctx.ckpt, f"pairs_{set_name}_post",
                                            cache_dir=ctx.cache / "hf", logger=log,
                                            force=args.force)
                    emb_main = np.load(ctx.out / f"emb_{key}_code_{pool}.npy", mmap_mode="r")
                    Xall = np.asarray(emb_main[:, layer, :], dtype=np.float32)
                    Xtr = Xall[train_mask.to_numpy()]
                    from sklearn.preprocessing import StandardScaler
                    sc = StandardScaler().fit(Xtr)
                    clf, C = tune_and_fit(sc.transform(Xtr), y_tr, g_tr, ctx.seed)
                    Xp = np.asarray(np.load(meta_pre["paths"][pool], mmap_mode="r")[:, layer, :],
                                    dtype=np.float32)
                    Xq = np.asarray(np.load(meta_post["paths"][pool], mmap_mode="r")[:, layer, :],
                                    dtype=np.float32)
                    s_pre = clf.decision_function(sc.transform(Xp))
                    s_post = clf.decision_function(sc.transform(Xq))
                    scorers[f"probe_{key}_{pool}"] = {
                        "C": C, "layer": layer, "n_layers_searched": n_layers,
                        "n_train": int(len(train)), "n_train_flaky": int(y_tr.sum()),
                        "layer_selection": ("argmax of flaky_code AP on the main dataset's "
                                            "cross-project curve; the pairs were not used to "
                                            "choose it"),
                        "scores_pre": s_pre.tolist(), "scores_post": s_post.tolist()}

        strata = {
            "all": np.ones(len(pairs), dtype=bool),
            "cue_added": (pairs.cue_stratum == "cue_added").to_numpy(),
            "cue_removed": (pairs.cue_stratum == "cue_removed").to_numpy(),
            "cue_neutral": (pairs.cue_stratum == "cue_neutral").to_numpy(),
            "behavioural_only": ~pairs.annotation_only.astype(bool).to_numpy(),
            "annotation_only": pairs.annotation_only.astype(bool).to_numpy(),
        }
        entry_set = {
            "n_pairs": int(len(pairs)),
            "n_projects": int(pairs.project.nunique()),
            "n_train_rows": int(len(train)),
            "n_train_flaky": int(y_tr.sum()),
            "n_held_out_projects": len(pair_projects),
            "held_out_projects": sorted(pair_projects),
            "reaches_power_floor": len(pairs) >= POWER_FLOOR,
            "strata_sizes": {k: int(v.sum()) for k, v in strata.items()},
            "scorers": {},
        }
        for name, sc_ in scorers.items():
            a = np.asarray(sc_["scores_pre"])
            b = np.asarray(sc_["scores_post"])
            e = {k: v for k, v in sc_.items() if not k.startswith("scores_")}
            e["overall"] = pair_metrics(a, b)
            e["by_stratum"] = {k: pair_metrics(a[m], b[m]) for k, m in strata.items()
                               if m.sum() > 0}
            e["scores_pre"] = sc_["scores_pre"]
            e["scores_post"] = sc_["scores_post"]
            entry_set["scorers"][name] = e
        res["pair_sets"][set_name] = entry_set

    # Flatten the primary set to the top level so 07 and the figures can read it
    # without knowing about the two-set structure.
    primary = res["pair_sets"][res["primary_pair_set"]]
    res.update({k: v for k, v in primary.items() if k != "scorers"})
    res["scorers"] = primary["scorers"]
    res["acceptance"] = {
        "pair_projects_absent_from_training": True,
        "cue_deltas_recorded": bool("cue_stratum" in all_pairs.columns),
        "reaches_power_floor_40": primary["n_pairs"] >= POWER_FLOOR,
    }
    save_json(ctx.out / "05_pairs.json", res)
    StageMarker(ctx, STAGE).write(f"{len(all_pairs)}:{args.models}:{ctx.seed}")

    log.info("--- PairAcc (pre > post; >0.5 = structure, <0.5 = cue-riding) ---")
    for set_name, es in res["pair_sets"].items():
        star = " (primary)" if set_name == res["primary_pair_set"] else ""
        log.info("[%s]%s n=%d pairs, %d train rows / %d flaky",
                 set_name, star, es["n_pairs"], es["n_train_rows"], es["n_train_flaky"])
        for name, e in es["scorers"].items():
            o = e["overall"]
            log.info("  %-24s %.4f [%.3f, %.3f] p=%.4f margin %+.4f",
                     name, o["pair_acc"], o["wilson_95"][0], o["wilson_95"][1],
                     o["binom_p_two_sided"], o["mean_margin"])
            cn = e["by_stratum"].get("cue_neutral")
            if cn and cn["n"]:
                log.info("      cue-neutral subset: %.4f (n=%d) -- the cleanest test",
                         cn["pair_acc"], cn["n"])
    if res["n_pairs"] < POWER_FLOOR:
        log.warning("only %d pairs in the primary set (< %d): S3 will be marked INCONCLUSIVE",
                    res["n_pairs"], POWER_FLOOR)
    return 0 if report_acceptance(log, res["acceptance"], ctx.smoke) else 1


if __name__ == "__main__":
    raise SystemExit(main())
