#!/usr/bin/env python
"""04 -- structural property labels.

Ground truth that does not depend on the flakiness label.  This is what makes
"structural information" a testable claim rather than a residue left over after
cue removal.

Emits ``data/structural.csv``, ``out/04_structural.json`` and the stratified
sample that the hand audit works from.  The audit is not optional: an
unvalidated static analyser is not ground truth, and ``s04_validate.py``
enforces the precision floor of 0.8 per retained property.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from flaky.common import (JsonlCheckpoint, StageMarker, Timer, add_common_args, ctx_from_args,
                          file_digest, get_logger, progress, report_acceptance, save_json,
                          save_text, set_seed)
from flaky.cv import fit_and_score, make_folds
from flaky.structural import PROPERTIES, analyse

STAGE = "04_structural"
AUDIT_N = 50


def build_audit_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Stratify by property and by label so the audit sees positives of every
    property and negatives of every class."""
    rng = np.random.RandomState(seed)
    picked: list[int] = []
    strata: list[tuple[str, pd.Index]] = []
    for p in PROPERTIES:
        strata.append((f"{p}=1&flaky", df.index[(df[p] == 1) & (df.flaky == 1)]))
        strata.append((f"{p}=1&non_flaky", df.index[(df[p] == 1) & (df.flaky == 0)]))
    none_mask = ~(df[list(PROPERTIES)].any(axis=1))
    strata.append(("none&flaky", df.index[none_mask & (df.flaky == 1)]))
    strata.append(("none&non_flaky", df.index[none_mask & (df.flaky == 0)]))

    per = max(1, n // len(strata))
    labels: dict[int, str] = {}
    for name, idx in strata:
        idx = np.asarray(idx)
        if len(idx) == 0:
            continue
        take = idx[rng.choice(len(idx), size=min(per, len(idx)), replace=False)]
        for i in take:
            if i not in labels:
                labels[int(i)] = name
                picked.append(int(i))
    # top up to n from the remaining rows, keeping determinism
    rest = np.setdiff1d(df.index.to_numpy(), np.asarray(picked))
    if len(picked) < n and len(rest):
        extra = rest[rng.choice(len(rest), size=min(n - len(picked), len(rest)), replace=False)]
        for i in extra:
            labels[int(i)] = "topup"
            picked.append(int(i))
    sample = df.loc[sorted(picked)[:n]].copy()
    sample["stratum"] = [labels[i] for i in sample.index]
    return sample


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--audit-n", type=int, default=AUDIT_N)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    ctx = ctx_from_args(args)
    log = get_logger(STAGE, ctx)
    set_seed(ctx.seed)

    prepped = ctx.data / "prepped.csv"
    df = pd.read_csv(prepped, keep_default_na=False, na_values=[])
    log.info("loaded %d rows", len(df))

    marker = StageMarker(ctx, STAGE)
    sig = f"{file_digest(prepped)}:{ctx.seed}:{args.audit_n}"
    if marker.matches(sig) and not args.force and (ctx.data / "structural.csv").exists():
        log.info("already done (signature %s); use --force to redo", sig)
        return 0

    # -- analyse (resumable) -------------------------------------------------
    cp = JsonlCheckpoint(ctx.ckpt / "04_structural.jsonl")
    if args.force:
        cp.reset()
    todo = [(i, r) for i, r in enumerate(df.itertuples()) if r.uid not in cp]
    if len(todo) < len(df):
        log.info("resuming: %d/%d rows already analysed", len(df) - len(todo), len(df))
    with Timer(log, f"tree-sitter analysis of {len(todo)} methods"), cp:
        for _, r in progress(todo, desc="structural", total=len(todo)):
            cp.put(r.uid, analyse(r.code))

    recs = [cp.get(u) for u in df["uid"]]
    for p in PROPERTIES:
        df[p] = [int(bool(r[p])) for r in recs]
    df["parse_ok"] = [int(bool(r["parse_ok"])) for r in recs]
    df["parse_repair"] = [r.get("parse_repair", "") or "" for r in recs]
    df["P_ASYNC_sleep_kills"] = [int(bool(r.get("P_ASYNC_sleep_kills", False))) for r in recs]
    witnesses = {u: r.get("witness", {}) for u, r in zip(df["uid"], recs)}

    parse_fail = float(1.0 - df["parse_ok"].mean())
    log.info("parse failure rate %.4f (acceptance < 0.05)", parse_fail)
    failed_ids = df.loc[df.parse_ok == 0, "id"].tolist()
    repair_counts = df.loc[df.parse_repair != "", "parse_repair"].value_counts().to_dict()
    log.info("rows needing a truncation repair: %d (%s)",
             int((df.parse_repair != "").sum()), repair_counts)

    struct = df[["uid", "id", *PROPERTIES, "parse_ok"]]
    struct.to_csv(ctx.data / "structural.csv", index=False)
    save_json(ctx.data / "structural_witnesses.json", witnesses)
    df[["uid", "id", "project", "test_name", "label", *PROPERTIES, "parse_ok", "parse_repair",
        "P_ASYNC_sleep_kills"]].to_csv(ctx.data / "structural_full.csv", index=False)

    # -- prevalence and cross-tabs ------------------------------------------
    res: dict = {
        "n_rows": int(len(df)),
        "parse_failure_rate": parse_fail,
        "parse_failed_ids": failed_ids[:200],
        "n_parse_failed": len(failed_ids),
        "prevalence_overall": {p: float(df[p].mean()) for p in PROPERTIES},
        "parse_repairs": {
            "n_repaired": int((df.parse_repair != "").sum()),
            "rate": float((df.parse_repair != "").mean()),
            "by_kind": repair_counts,
            "failure_rate_without_repair": float(
                1.0 - ((df.parse_ok == 1) & (df.parse_repair == "")).mean()),
            "note": ("FlakeBench truncates a fraction of method bodies mid-statement. Those "
                     "rows are 99% non-flaky, so leaving them unparsed would bias the labels "
                     "in exactly the direction the plan warns about. The repair closes open "
                     "brackets or rewinds to the last complete statement; it never invents "
                     "content. Every repaired row is flagged in data/structural_full.csv."),
        },
        "parse_failures_by_label": df.loc[df.parse_ok == 0, "label"].value_counts().to_dict(),
        "P_ASYNC_definition_diagnostic": {
            "P_ASYNC_rate": float(df["P_ASYNC"].mean()),
            "P_ASYNC_rate_if_sleep_were_not_a_barrier": float(df["P_ASYNC_sleep_kills"].mean()),
            "n_rows_that_would_flip": int(
                ((df.P_ASYNC_sleep_kills == 1) & (df.P_ASYNC == 0)).sum()),
            "rate_in_async_wait_if_sleep_not_a_barrier": float(
                df.loc[df.label == "async wait", "P_ASYNC_sleep_kills"].mean())
            if (df.label == "async wait").any() else None,
            "note": ("Diagnostic only -- not a property, not in any stopping rule. The plan "
                     "puts `sleep` in the synchronisation kill set, which excludes the "
                     "canonical async-wait flaky shape (dispatch; sleep; assert) from P_ASYNC "
                     "by construction. This quantifies how much of the category that removes."),
        },
        "prevalence_by_label": {},
        "crosstab_vs_flaky": {},
        "not_implemented": {
            "P_SHARED_FIELD": ("Shared field written by one test and read by another is "
                               "inter-procedural: it needs the class body and sibling tests "
                               "and cannot be computed from FlakeBench method bodies. "
                               "Deferred to Phase 2, which requires repository checkout."),
        },
    }
    for lab, sub in df.groupby("label"):
        res["prevalence_by_label"][lab] = {"n": int(len(sub)),
                                           **{p: float(sub[p].mean()) for p in PROPERTIES}}
    for p in PROPERTIES:
        ct = pd.crosstab(df[p], df["flaky"])
        res["crosstab_vs_flaky"][p] = {str(k): {str(kk): int(vv) for kk, vv in v.items()}
                                       for k, v in ct.to_dict().items()}

    # -- do the structural properties predict flakiness? --------------------
    y = df["flaky"].to_numpy()
    groups = df["project"].to_numpy()
    X = df[list(PROPERTIES)].to_numpy(dtype=np.float64)
    folds = make_folds(y, groups, "cross", n_splits=args.folds, seed=ctx.seed)
    fold_res = []
    for tr, te in folds:
        r, _ = fit_and_score(X[tr], y[tr], groups[tr], X[te], y[te], "cross", seed=ctx.seed)
        fold_res.append(r)
    aps = [r["ap"] for r in fold_res]
    res["structural_only_classifier"] = {
        "regime": "cross", "features": list(PROPERTIES),
        "ap": {"mean": float(np.mean(aps)), "std": float(np.std(aps, ddof=1)),
               "folds": aps},
        "auroc_folds": [r["auroc"] for r in fold_res],
        "class_prior": float(y.mean()),
        "note": ("If the three booleans carry real signal the properties are meaningful; "
                 "if not, either the analysis is weak or flakiness in this dataset is not "
                 "captured by them. Either way it is reportable."),
    }
    log.info("structural-only cross-project AP = %.4f +- %.4f (prior %.4f)",
             np.mean(aps), np.std(aps, ddof=1), y.mean())

    # -- floor check ---------------------------------------------------------
    async_rate = float(df.loc[df.label == "async wait", "P_ASYNC"].mean()) \
        if (df.label == "async wait").any() else float("nan")
    nonflaky_rate = float(df.loc[df.label == "non-flaky", "P_ASYNC"].mean())
    res["floor_check"] = {
        "P_ASYNC_rate_async_wait": async_rate,
        "P_ASYNC_rate_non_flaky": nonflaky_rate,
        "passes": bool(async_rate > nonflaky_rate),
        "note": "A floor check, not a finding. If it fails the analysis is broken.",
    }
    log.info("floor check: P_ASYNC in 'async wait' %.3f vs non-flaky %.3f -> %s",
             async_rate, nonflaky_rate, "PASS" if async_rate > nonflaky_rate else "FAIL")

    # -- audit sample --------------------------------------------------------
    sample = build_audit_sample(df, args.audit_n, ctx.seed)
    lines = ["# 04 -- hand-audit sample",
             "",
             f"{len(sample)} methods, stratified by property and by label.",
             "",
             "For each, decide by reading the code whether each property holds, using the",
             "definitions in `plan/04_STRUCTURAL_LABELS.md`, then record the judgement in",
             "`data/04_audit_labels.json` as `{\"<id>\": {\"P_ASYNC\": 0|1, ...}}`.",
             "",
             "The analyser's own prediction is shown so the audit can be checked, not so it",
             "can be copied. Judge from the code.",
             ""]
    for r in sample.itertuples():
        pred = {p: int(getattr(r, p)) for p in PROPERTIES}
        w = witnesses.get(r.uid, {})
        lines += [f"## uid {r.uid}  (id {r.id}, {r.stratum})",
                  "",
                  f"- project: `{r.project}`",
                  f"- test: `{r.test_name}`",
                  f"- label: `{r.label}`   parse_ok: {int(r.parse_ok)}",
                  f"- analyser: {pred}",
                  f"- witness: {({k: v[:120] for k, v in w.items()}) if w else '{}'}",
                  "",
                  "```java",
                  r.code,
                  "```",
                  ""]
    save_text(ctx.out / "04_audit_sample.md", "\n".join(lines))
    save_json(ctx.data / "04_audit_template.json",
              {r.uid: {p: None for p in PROPERTIES} for r in sample.itertuples()})
    res["audit_sample_ids"] = sample["uid"].tolist()
    res["audit_sample_strata"] = sample["stratum"].value_counts().to_dict()

    res["acceptance"] = {
        "parse_failure_rate_under_5pct": parse_fail < 0.05,
        "floor_check_P_ASYNC": res["floor_check"]["passes"],
        "audit_emitted": True,
        "precision_floor": "checked by scripts/s04_validate.py against the hand audit",
    }
    save_json(ctx.out / "04_structural.json", res)
    marker.write(sig)

    log.info("prevalence: %s", {p: round(res["prevalence_overall"][p], 4) for p in PROPERTIES})
    log.info("audit sample written to %s (%d methods)",
             ctx.out / "04_audit_sample.md", len(sample))
    return 0 if report_acceptance(log, res["acceptance"], ctx.smoke) else 1


if __name__ == "__main__":
    raise SystemExit(main())
