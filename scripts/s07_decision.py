#!/usr/bin/env python
"""07 -- stopping rule and write-up.

The rule was fixed in ``plan/00_OVERVIEW.md`` before any results were seen.  No
threshold is adjusted here, no baseline is swapped, no metric is added.

S1  cue-removed probe beats ``STRONG_WORD_BASELINE``, cross-project.  Paired
    one-sided Wilcoxon over the 5 folds, alpha = 0.05.
S2  ``delta_probe < delta_bow``, paired across folds.
S3  minimal-pair accuracy above 0.5 for the probe, binomial, alpha = 0.05;
    inconclusive rather than failed below 40 usable pairs.

**S1's operationalisation.** "Cue-removed probe" is read as the probe evaluated
on cue-removed input: trained on ``code``, scored on ``code_renamed``, against a
word baseline that had full access to the cues.  That is the Phase 1 question
in 00 -- *after removing lexical cues, does any predictive signal survive
across unseen projects* -- and it is the framing reported as primary.  Two other
readings exist (probe on ``code``; probe trained and evaluated on
``code_renamed``), and both are computed and reported here so a reader can see
the verdict under each rather than take the primary framing on trust.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from flaky.common import (add_common_args, ctx_from_args, get_logger, load_json, save_json,
                          save_text)
from flaky.stats import wilcoxon_one_sided
from flaky.structural import PROPERTIES

STAGE = "07_decision"
ALPHA = 0.05
PAIR_POWER_FLOOR = 40

S1_FRAMINGS = {
    "primary_cue_removed": ("probe trained on `code`, evaluated on `code_renamed`",
                            "ap_on_renamed_by_layer"),
    "secondary_on_code": ("probe trained and evaluated on `code`", "ap_by_layer"),
    "secondary_retrained_renamed": ("probe trained and evaluated on `code_renamed`", None),
}


def pick_probe(probe: dict) -> tuple[str, str, dict]:
    """The probe the rule is evaluated on: the best (model, pool) by
    cross-project AP of `flaky_code`, chosen before looking at any renamed or
    pair result."""
    best = None
    for key, pools in probe.get("models", {}).items():
        for pool, entry in pools.items():
            ap = entry["targets"]["flaky_code"]["max"]["ap_mean"]
            if best is None or ap > best[2]:
                best = (key, pool, ap)
    if best is None:
        raise SystemExit("06_probe.json has no probed model")
    key, pool, _ = best
    return key, pool, probe["models"][key][pool]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    args = ap.parse_args()

    ctx = ctx_from_args(args)
    log = get_logger(STAGE, ctx)

    need = ["01_prep.json", "02_baselines.json", "03_transfer.json", "04_structural.json",
            "06_probe.json"]
    missing = [n for n in need if not (ctx.out / n).exists()]
    if missing:
        log.error("missing result files: %s", missing)
        return 1

    prep = load_json(ctx.out / "01_prep.json")
    base = load_json(ctx.out / "02_baselines.json")
    transfer = load_json(ctx.out / "03_transfer.json")
    struct = load_json(ctx.out / "04_structural.json")
    probe = load_json(ctx.out / "06_probe.json")
    valid = load_json(ctx.out / "04_validation.json") \
        if (ctx.out / "04_validation.json").exists() else None
    pairs = load_json(ctx.out / "05_pairs.json") \
        if (ctx.out / "05_pairs.json").exists() else None

    swb = base["STRONG_WORD_BASELINE"]
    key, pool, entry = pick_probe(probe)
    layer = entry["targets"]["flaky_code"]["max"]["layer"]
    n_layers = entry["n_layers"]
    log.info("probe under test: %s/%s, layer %d of %d (chosen by cross-project AP on `code`)",
             key, pool, layer, n_layers)

    verdict: dict = {
        "alpha": ALPHA,
        "probe": {"model": key, "pool": pool, "layer": layer, "n_layers_searched": n_layers,
                  "selection": ("argmax of flaky_code cross-project AP; a maximum over a "
                                "searched curve, not a held-out estimate")},
        "strong_word_baseline": swb,
        "criteria": {},
    }

    # ------------------------------------------------------------------ S1
    swb_folds = np.asarray(swb["ap_folds"], dtype=np.float64)
    t = entry["targets"]["flaky_code"]
    framings = {
        "primary_cue_removed": np.asarray(t["ap_on_renamed_by_layer"][layer]["folds"]),
        "secondary_on_code": np.asarray(t["ap_by_layer"][layer]["folds"]),
    }
    if "flaky_renamed" in entry["targets"]:
        fr = entry["targets"]["flaky_renamed"]["ap_by_layer"][layer]
        framings["secondary_retrained_renamed"] = np.asarray(fr["folds"])

    s1: dict = {"description": S1_FRAMINGS["primary_cue_removed"][0],
                "baseline_method": swb["method"], "baseline_folds": swb_folds.tolist(),
                "framings": {}}
    for name, folds in framings.items():
        w = wilcoxon_one_sided(folds, swb_folds, alternative="greater")
        w["probe_ap_mean"] = float(folds.mean())
        w["baseline_ap_mean"] = float(swb_folds.mean())
        w["passes"] = bool(w.get("p", 1.0) < ALPHA and w["mean_diff"] > 0)
        w["framing"] = S1_FRAMINGS[name][0]
        s1["framings"][name] = w
    s1["passes"] = s1["framings"]["primary_cue_removed"]["passes"]
    s1["note"] = ("Five folds: the smallest attainable one-sided p is 1/32 = 0.031, so a "
                  "pass requires every fold to favour the probe. Fold-level differences are "
                  "listed in full above the verdict.")
    verdict["criteria"]["S1"] = s1

    # ------------------------------------------------------------------ S2
    bow_delta = np.asarray(transfer["methods"]["bow"]["delta"]["folds"], dtype=np.float64)
    probe_delta = np.asarray(t["transfer_delta_by_layer"][layer]["folds"], dtype=np.float64)
    w2 = wilcoxon_one_sided(bow_delta, probe_delta, alternative="greater")
    s2 = {
        "description": "renaming transfer drop is smaller for the probe than for the word baseline",
        "delta_probe_folds": probe_delta.tolist(),
        "delta_bow_folds": bow_delta.tolist(),
        "delta_probe_mean": float(probe_delta.mean()),
        "delta_bow_mean": float(bow_delta.mean()),
        "paired_test_bow_minus_probe": w2,
        "passes": bool(probe_delta.mean() < bow_delta.mean()),
        "passes_paired_majority": bool((probe_delta < bow_delta).sum() > len(bow_delta) / 2),
        "n_folds_probe_smaller": int((probe_delta < bow_delta).sum()),
    }
    verdict["criteria"]["S2"] = s2

    # ------------------------------------------------------------------ S3
    if pairs is None or pairs.get("n_pairs", 0) == 0:
        s3 = {"description": "minimal-pair accuracy above 0.5 for the probe",
              "status": "inconclusive", "passes": None,
              "reason": "no minimal pairs available"}
    else:
        pname = f"probe_{key}_{pool}"
        sc = pairs["scorers"].get(pname) or next(
            (v for k, v in pairs["scorers"].items() if k.startswith("probe_")), None)
        bow = pairs["scorers"].get("bow")
        n = pairs["n_pairs"]
        if sc is None:
            s3 = {"status": "inconclusive", "passes": None,
                  "reason": "no probe scorer in 05_pairs.json"}
        else:
            o = sc["overall"]
            below_power = n < PAIR_POWER_FLOOR
            s3 = {
                "description": "minimal-pair accuracy above 0.5 for the probe",
                "scorer": pname,
                "n_pairs": n,
                "pair_acc": o["pair_acc"],
                "wilson_95": o["wilson_95"],
                "binom_p_two_sided": o["binom_p_two_sided"],
                "mean_margin": o["mean_margin"],
                "bow_pair_acc": bow["overall"]["pair_acc"] if bow else None,
                "bow_mean_margin": bow["overall"]["mean_margin"] if bow else None,
                "cue_neutral": sc["by_stratum"].get("cue_neutral"),
                "bow_cue_neutral": bow["by_stratum"].get("cue_neutral") if bow else None,
                "status": "inconclusive" if below_power else "evaluated",
                "passes": None if below_power else bool(
                    o["pair_acc"] > 0.5 and o["binom_p_two_sided"] < ALPHA),
                "power_note": (f"{n} usable pairs; below {PAIR_POWER_FLOOR} the binomial test "
                               "has too little power and S3 is inconclusive, not failed."),
                "sign_note": ("The contrast with bow is the substantive result: a cue-riding "
                              "model is expected below 0.5 because fixes add cue tokens."),
            }
    verdict["criteria"]["S3"] = s3

    # ------------------------------------------------------------- overall
    passes = [verdict["criteria"][k].get("passes") for k in ("S1", "S2", "S3")]
    if all(p is True for p in passes):
        overall = "proceed_to_phase_2"
    elif any(p is None for p in passes) and not any(p is False for p in passes):
        overall = "inconclusive"
    else:
        overall = "stop_and_write_up"
    verdict["overall"] = overall
    verdict["overall_note"] = (
        "All three must hold to proceed to Phase 2. Otherwise stop and write up. "
        "The negative write-up is a deliverable, not a fallback.")

    # possible split verdict: structure encoded but not predictive of the label
    prop_ap = {p: entry["targets"][p]["max"]["ap_mean"] for p in PROPERTIES
               if p in entry["targets"]}
    prop_sel = {p: entry["targets"][p]["selectivity_by_layer"][
        entry["targets"][p]["max"]["layer"]]["mean"] for p in PROPERTIES
        if p in entry["targets"]}
    verdict["split_verdict"] = {
        "probe_encodes_properties_ap": prop_ap,
        "probe_property_selectivity": prop_sel,
        "structural_only_classifier_ap": struct["structural_only_classifier"]["ap"]["mean"],
        "class_prior": struct["structural_only_classifier"]["class_prior"],
        "note": ("The probe may fail S1 while still encoding P_ASYNC well. That would mean "
                 "the model represents the structure but the structure does not predict this "
                 "dataset's labels -- a different and more interesting finding than a flat "
                 "negative, and it is reported as such."),
    }

    save_json(ctx.out / "07_decision.json", verdict)

    # ------------------------------------------------------------- the report
    from report import build_report  # noqa: E402  (local module, same directory)

    md = build_report(ctx, prep, base, transfer, struct, valid, pairs, probe, verdict,
                      key, pool, layer)
    save_text(ctx.out / "07_report.md", md)

    log.info("=" * 66)
    for k in ("S1", "S2", "S3"):
        c = verdict["criteria"][k]
        p = c.get("passes")
        log.info("%s: %s -- %s", k, {True: "PASS", False: "FAIL", None: "INCONCLUSIVE"}[p],
                 c.get("description", ""))
    log.info("OVERALL: %s", overall.upper())
    log.info("report written to %s", ctx.out / "07_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
