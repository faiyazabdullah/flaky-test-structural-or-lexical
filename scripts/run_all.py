#!/usr/bin/env python
"""Run the whole pipeline, resumably.

Stage order follows the plan's dependencies, which differ from its file
numbering in one place: ``03`` builds the cue vocabulary and the renamed
variant, and ``02`` needs both (``bow_ablated`` is ``bow`` with the cue
vocabulary removed, and the transfer measurement reuses ``02``'s fitted
models).  So ``03`` runs before ``02``.  ``05`` splits in two: mining needs only
the corpus, scoring needs the probe from ``06``.

Each stage writes ``ckpt/<stage>.done`` recording the inputs it saw; a re-run
skips a stage whose inputs are unchanged.  Long loops inside a stage keep their
own JSONL checkpoints, so a kill mid-stage resumes at the next item rather than
the next stage.

    python scripts/run_all.py --smoke     # ~10 min end to end, validates plumbing
    python scripts/run_all.py             # the real run
    python scripts/run_all.py --from 06   # resume from a stage
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flaky.common import Ctx, fmt_dur, get_logger, save_json

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

# (key, script, extra args, description, required)
STAGES = [
    ("01", "s01_prep.py", [], "data preparation and the whitespace audit", True),
    ("03", "s03_cues_rename.py", [], "cue vocabulary and counterfactual renaming", True),
    ("02", "s02_baselines.py", [], "baselines, within-vs-cross gap, transfer drop", True),
    ("04", "s04_structural.py", [], "structural property labels + audit sample", True),
    ("05a", "s05_mine_pairs.py", [], "mine minimal pairs from fix commits", False),
    ("06", "s06_probe.py", [], "hidden-state extraction and linear probes", True),
    ("04v", "s04_validate.py", [], "score the analyser against the hand audit", False),
    ("05b", "s05_score_pairs.py", [], "score the minimal pairs", False),
    ("07", "s07_decision.py", [], "stopping rule and write-up", True),
    ("fig", "figures.py", [], "figures", False),
    ("md", "export_reports.py", [], "collect every result into reports/ as Markdown", False),
]


def run_stage(key: str, script: str, extra: list[str], args, log) -> int:
    cmd = [PY, str(ROOT / "scripts" / script), "--seed", str(args.seed), *extra]
    if args.smoke:
        cmd.append("--smoke")
    if args.force:
        cmd.append("--force")
    if key == "06" and args.models:
        cmd += ["--models", args.models]
    if key == "05b" and args.models:
        cmd += ["--models", args.models]
    if key == "05a" and args.max_commits:
        cmd += ["--max-commits", str(args.max_commits)]
    log.info("$ %s", " ".join(cmd))
    t0 = time.monotonic()
    rc = subprocess.call(cmd, cwd=str(ROOT))
    log.info("stage %s exited %d after %s", key, rc, fmt_dur(time.monotonic() - t0))
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true",
                    help="reduced subset, isolated data_smoke/ and out_smoke/ trees")
    ap.add_argument("--force", action="store_true", help="ignore checkpoints")
    ap.add_argument("--from", dest="from_stage", default=None,
                    help=("start at this stage key "
                          "(01, 03, 02, 04, 05a, 06, 04v, 05b, 07, fig, md)"))
    ap.add_argument("--only", default=None, help="comma-separated stage keys to run")
    ap.add_argument("--models", default=None, help="passed to 06 and 05b")
    ap.add_argument("--max-commits", type=int, default=None, help="passed to 05a")
    ap.add_argument("--tests", action="store_true", help="run the unit tests first")
    ap.add_argument("--skip-tests", action="store_true")
    args = ap.parse_args()

    ctx = Ctx(smoke=args.smoke, seed=args.seed)
    log = get_logger("run_all", ctx)
    log.info("mode: %s, seed %d, root %s", "SMOKE" if args.smoke else "FULL", args.seed, ROOT)

    if args.tests or not args.skip_tests:
        log.info("running unit tests (renaming and structural analysis are where a silent "
                 "bug invalidates everything downstream)")
        rc = subprocess.call([PY, "-m", "pytest", "tests/", "-q"], cwd=str(ROOT))
        if rc != 0:
            log.error("unit tests failed -- refusing to run the pipeline")
            return rc

    stages = STAGES
    if args.only:
        keep = {s.strip() for s in args.only.split(",")}
        stages = [s for s in STAGES if s[0] in keep]
    elif args.from_stage:
        keys = [s[0] for s in STAGES]
        if args.from_stage not in keys:
            log.error("unknown stage %r; known: %s", args.from_stage, keys)
            return 2
        stages = STAGES[keys.index(args.from_stage):]

    summary = {"mode": "smoke" if args.smoke else "full", "seed": args.seed, "stages": {}}
    t0 = time.monotonic()
    for key, script, extra, desc, required in stages:
        log.info("=" * 72)
        log.info("STAGE %s -- %s", key, desc)
        t = time.monotonic()
        rc = run_stage(key, script, extra, args, log)
        summary["stages"][key] = {"script": script, "rc": rc,
                                  "seconds": round(time.monotonic() - t, 1),
                                  "description": desc}
        save_json(ctx.out / "run_all_summary.json", summary)
        if rc != 0:
            if required:
                log.error("required stage %s failed (rc=%d) -- stopping", key, rc)
                return rc
            log.warning("optional stage %s failed (rc=%d) -- continuing", key, rc)

    log.info("=" * 72)
    log.info("pipeline finished in %s", fmt_dur(time.monotonic() - t0))
    for k, v in summary["stages"].items():
        log.info("  %-4s %-24s rc=%d  %s", k, v["script"], v["rc"], fmt_dur(v["seconds"]))
    log.info("report:  %s", ctx.out / "07_report.md")
    log.info("markdown bundle: %s", ctx.root / f"reports{ctx.suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
