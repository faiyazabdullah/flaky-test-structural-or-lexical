#!/usr/bin/env python
"""04 (validation) -- score the static analyser against the hand audit.

``s04_structural.py`` writes ``out/04_audit_sample.md`` (50 methods, stratified
by property and by label) and ``data/04_audit_template.json``.  The auditor
reads each method, decides by the definitions in ``plan/04`` whether each
property holds, and fills the judgements into ``data/04_audit_labels.json``.
This script turns those judgements into precision and recall.

Precision below 0.8 on any property means that property is not ground truth:
either the analysis is fixed or the property is dropped from the study.  An
unvalidated static analyser is not ground truth.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from flaky.common import (add_common_args, ctx_from_args, get_logger, load_json,
                          report_acceptance, save_json, save_text)
from flaky.stats import wilson_interval
from flaky.structural import PROPERTIES

STAGE = "04_validate"
PRECISION_FLOOR = 0.8


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--labels", default=None,
                    help="auditor's judgements (default data/04_audit_labels.json)")
    ap.add_argument("--auditor", default="Claude (implementing agent)",
                    help="who performed the audit -- recorded verbatim in the report")
    args = ap.parse_args()

    ctx = ctx_from_args(args)
    log = get_logger(STAGE, ctx)

    labels_path = Path(args.labels) if args.labels else ctx.data / "04_audit_labels.json"
    if not labels_path.exists():
        log.error("no audit labels at %s. Read out/04_audit_sample.md, fill in "
                  "data/04_audit_template.json (keyed by uid) and save it as "
                  "04_audit_labels.json.", labels_path)
        return 1

    truth = load_json(labels_path)
    provenance = truth.pop("_provenance", {})
    if provenance.get("auditor"):
        args.auditor = provenance["auditor"]
    struct = pd.read_csv(ctx.data / "structural_full.csv").set_index("uid")

    ids = [i for i in truth if i in struct.index]
    missing = [i for i in truth if i not in struct.index]
    if missing:
        log.warning("%d audited ids are not in structural_full.csv: %s", len(missing), missing[:5])
    log.info("audit covers %d methods", len(ids))

    res: dict = {"n_audited": len(ids), "auditor": args.auditor,
                 "precision_floor": PRECISION_FLOOR, "provenance": provenance,
                 "properties": {}, "semantic": {}}
    rows_md = []

    # Second scoring against the semantic judgement, where present: does the
    # property hold in the code's actual behaviour, not merely by the
    # definition's fixed vocabulary?  Reported, never used as the acceptance
    # gate -- the acceptance gate is the definition, which is what the labels
    # claim to encode.
    for p in PROPERTIES:
        key = f"{p}_semantic"
        if not all(key in truth[i] for i in ids):
            continue
        yt = np.asarray([int(bool(truth[i][key])) for i in ids])
        yp = np.asarray([int(struct.loc[i, p]) for i in ids])
        tp = int(((yp == 1) & (yt == 1)).sum())
        fp = int(((yp == 1) & (yt == 0)).sum())
        fn = int(((yp == 0) & (yt == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        rec = tp / (tp + fn) if (tp + fn) else float("nan")
        res["semantic"][p] = {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec}

    for p in PROPERTIES:
        y_true = np.asarray([int(bool(truth[i][p])) for i in ids])
        y_pred = np.asarray([int(struct.loc[i, p]) for i in ids])
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        rec = tp / (tp + fn) if (tp + fn) else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if (tp + fp) and (tp + fn) and (prec + rec) else \
            float("nan")
        plo, phi = wilson_interval(tp, tp + fp) if (tp + fp) else (float("nan"), float("nan"))
        rlo, rhi = wilson_interval(tp, tp + fn) if (tp + fn) else (float("nan"), float("nan"))
        entry = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec,
                 "f1": f1, "precision_wilson_95": [plo, phi], "recall_wilson_95": [rlo, rhi],
                 "n_predicted_positive": tp + fp, "n_true_positive": tp + fn,
                 "meets_precision_floor": bool(not np.isnan(prec) and prec >= PRECISION_FLOOR),
                 "false_positive_ids": [i for i, a, b in zip(ids, y_pred, y_true)
                                        if a == 1 and b == 0],
                 "false_negative_ids": [i for i, a, b in zip(ids, y_pred, y_true)
                                        if a == 0 and b == 1]}
        res["properties"][p] = entry
        log.info("%-8s precision %.3f [%.2f, %.2f] (%d/%d)  recall %.3f (%d/%d)  -> %s",
                 p, prec, plo, phi, tp, tp + fp, rec, tp, tp + fn,
                 "KEEP" if entry["meets_precision_floor"] else "BELOW FLOOR")
        rows_md.append(
            f"| `{p}` | {tp} | {fp} | {fn} | {tn} | "
            f"{prec:.3f} [{plo:.2f}, {phi:.2f}] | {rec:.3f} [{rlo:.2f}, {rhi:.2f}] | "
            f"{'keep' if entry['meets_precision_floor'] else '**below floor**'} |")

    retained = [p for p in PROPERTIES if res["properties"][p]["meets_precision_floor"]]
    res["retained_properties"] = retained
    res["dropped_properties"] = [p for p in PROPERTIES if p not in retained]
    res["acceptance"] = {
        "precision_floor_met_for_all_properties": len(retained) == len(PROPERTIES),
        "audit_size_at_least_50": len(ids) >= 50,
    }

    md = [
        "# 04 — validation of the structural analyser",
        "",
        f"- audited methods: **{len(ids)}**, stratified by property and by label",
        f"- auditor: **{args.auditor}**",
        f"- precision floor: **{PRECISION_FLOOR}** — below it a property is either fixed or "
        "dropped from the study",
        "",
        "| property | TP | FP | FN | TN | precision (95% Wilson) | recall (95% Wilson) | verdict |",
        "|---|---|---|---|---|---|---|---|",
        *rows_md,
        "",
        "### Definitional vs semantic",
        "",
        "The table above scores the analyser against the properties **as written** in the",
        "plan — with its fixed vocabularies, so `execute` is an async dispatch and",
        "`currentTimeMillis` is a clock read wherever they appear. That is the acceptance",
        "gate, because it is what the labels claim to encode.",
        "",
        "A second judgement asks whether each property holds in the code's *actual*",
        "behaviour. The two diverge wherever the vocabulary matches a name that is not",
        "doing what the name suggests: Hystrix's `cmd.execute()` is the **synchronous**",
        "API, `SystemCtl.start(\"docker\")` builds a command object, and a",
        "`currentTimeMillis()` used only to name a temp directory never influences what is",
        "asserted.",
        "",
        "| property | semantic precision | semantic recall |",
        "|---|---|---|",
        *[f"| `{p}` | {res['semantic'][p]['precision']:.3f} "
          f"({res['semantic'][p]['tp']}/{res['semantic'][p]['tp'] + res['semantic'][p]['fp']}) "
          f"| {res['semantic'][p]['recall']:.3f} |"
          for p in PROPERTIES if p in res["semantic"]],
        "",
        "The gap is a property of the plan's definitions, not a defect in the",
        "implementation, and it bounds how much a probe trained on these labels can be",
        "said to have learned about genuine asynchrony or genuine order-dependence.",
        "",
        "## What the audit judged",
        "",
        "Each method was read against the definitions in `plan/04_STRUCTURAL_LABELS.md`:",
        "",
        "- `P_ASYNC` — an async dispatch (`submit`, `execute`, `start`, `thenApply`,",
        "  `runAsync`, `supplyAsync`, `schedule`) reaches an assertion with no intervening",
        "  synchronisation (`get`, `join`, `await`, `sleep`, `invokeAll`) on the path.",
        "- `P_UNORD` — a value originating from unordered iteration reaches an assertion",
        "  argument.",
        "- `P_CLOCK` — a clock read reaches an assertion argument.",
        "",
        "## Limitations of this audit",
        "",
        f"- The audit was performed by {args.auditor}, not by an independent human expert.",
        *([f"- {provenance['development_pass']}"] if provenance.get("development_pass") else []),
        "  It is a real reading of each method against the written definitions, but it is not",
        "  an inter-rater study and carries no second opinion.",
        "- 50 methods bound the precision estimate loosely; the Wilson intervals above are the",
        "  honest width.",
        "- Recall is measured only against the audited sample, which is stratified towards",
        "  predicted positives and therefore *over*-represents them relative to the corpus.",
        "",
        "## Errors",
        "",
    ]
    for p in PROPERTIES:
        e = res["properties"][p]
        md.append(f"- `{p}` false positives: {e['false_positive_ids'] or 'none'}")
        md.append(f"- `{p}` false negatives: {e['false_negative_ids'] or 'none'}")
    md.append("")

    save_text(ctx.out / "04_validation.md", "\n".join(md))
    save_json(ctx.out / "04_validation.json", res)
    log.info("wrote %s", ctx.out / "04_validation.md")
    return 0 if report_acceptance(log, res["acceptance"], ctx.smoke) else 1


if __name__ == "__main__":
    raise SystemExit(main())
