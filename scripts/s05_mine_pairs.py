#!/usr/bin/env python
"""05 (part 1) -- mine minimal pairs from fix commits.

The converse of the renaming test.  Renaming holds behaviour fixed and varies
surface; this holds surface nearly fixed and varies behaviour: ``x_pre`` is a
test method as it stood before a developer's flakiness fix, ``x_post`` the same
method after.

Commits come from ``dataset/filtered_tests_with_owner_sha.csv``.  Each is
fetched with ``--depth=2 --filter=blob:none``, so mining a repo the size of
hadoop costs a few hundred KB rather than a clone, and every fetch is cached
under ``cache/git/``.  Progress is checkpointed per commit: a killed run
resumes at the next one.

Excluded pairs are persisted with reasons.  An honest count of what did not
qualify is part of the result, not an inconvenience.
"""
from __future__ import annotations

import argparse
import csv
import difflib
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from flaky.common import (JsonlCheckpoint, StageMarker, Timer, add_common_args, ctx_from_args,
                          get_logger, load_json, progress, save_json, set_seed)
from flaky.gitfetch import GitError, changed_files, commit_meta, fetch_commit, file_at
from flaky.javalex import edit_distance, identifiers, token_texts
from flaky.javamethods import extract_methods, is_test_path
from flaky.normalize import normalize_code

STAGE = "05_mine_pairs"
MAX_DIFF_FRAC = 0.40      # above this the pair is no longer minimal
EDIT_DISTANCE_CAP = 3000  # above this, use a bag difference instead of Levenshtein


def token_diff(a: list[str], b: list[str]) -> tuple[int, bool]:
    """Token edit distance, with a cheap fallback for very long methods."""
    if max(len(a), len(b)) > EDIT_DISTANCE_CAP:
        ca, cb = Counter(a), Counter(b)
        return int(sum(((ca - cb) + (cb - ca)).values())), True
    return edit_distance(a, b), False


def cue_delta(pre: str, post: str, cues: set[str]) -> dict:
    ca, cb = Counter(i for i in identifiers(pre) if i in cues), \
        Counter(i for i in identifiers(post) if i in cues)
    added = {k: v for k, v in (cb - ca).items()}
    removed = {k: v for k, v in (ca - cb).items()}
    n_add, n_rem = sum(added.values()), sum(removed.values())
    if n_add > n_rem:
        stratum = "cue_added"
    elif n_rem > n_add:
        stratum = "cue_removed"
    else:
        stratum = "cue_neutral"
    return {"cue_tokens_added": added, "cue_tokens_removed": removed,
            "n_cue_added": n_add, "n_cue_removed": n_rem, "cue_stratum": stratum}


def annotation_only_change(pre: str, post: str) -> bool:
    """True when the only difference is annotations -- ``@Ignore`` on a flaky
    test is a 'fix' that changes no behaviour, and the write-up should say how
    many pairs are of that kind."""
    def strip_annotations(src: str) -> str:
        return "\n".join(l for l in src.splitlines() if not l.strip().startswith("@"))
    return strip_annotations(pre).strip() == strip_annotations(post).strip()


def mine_commit(cache: Path, owner_repo: str, sha: str, targets: dict, cues: set[str],
                log) -> dict:
    """All candidate pairs from one commit.  Never raises: failures are data."""
    out = {"owner_repo": owner_repo, "sha": sha, "pairs": [], "excluded": [], "error": None}
    try:
        repo = fetch_commit(cache, owner_repo, sha)
        meta = commit_meta(repo, sha)
        files = changed_files(repo, sha)
    except (GitError, subprocess.TimeoutExpired, OSError) as exc:
        out["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        return out

    out["subject"] = meta.get("subject", "")
    test_files = [(s, p) for s, p in files if is_test_path(p)]
    out["n_changed_files"] = len(files)
    out["n_changed_test_files"] = len(test_files)
    if not test_files:
        out["excluded"].append({"reason": "no_test_file_changed", "sha": sha,
                                "n_changed_files": len(files)})
        return out

    for status, path in test_files:
        post_src = file_at(repo, sha, path)
        pre_src = file_at(repo, f"{sha}^", path)
        if post_src is None or pre_src is None:
            out["excluded"].append({"reason": "file_missing_on_one_side", "path": path,
                                    "sha": sha})
            continue
        try:
            m_pre = extract_methods(pre_src)
            m_post = extract_methods(post_src)
        except Exception as exc:  # pragma: no cover
            out["excluded"].append({"reason": f"parse_error:{type(exc).__name__}",
                                    "path": path, "sha": sha})
            continue

        for key in sorted(set(m_pre) | set(m_post)):
            if "." not in key:
                continue  # bare-name keys duplicate the qualified ones
            tgt = targets.get(key)
            in_flakebench = tgt is not None
            if key not in m_pre or key not in m_post:
                if in_flakebench:
                    out["excluded"].append({"reason": "method_missing_on_one_side",
                                            "test": key, "path": path, "sha": sha})
                continue
            pre = normalize_code(m_pre[key])
            post = normalize_code(m_post[key])
            if pre == post:
                continue  # untouched by this commit
            if not (in_flakebench or key.split(".")[-1].lower().startswith("test")
                    or "@Test" in m_pre[key]):
                continue  # not a test method

            ta, tb = token_texts(pre), token_texts(post)
            dist, approx = token_diff(ta, tb)
            denom = max(len(ta), len(tb), 1)
            frac = dist / denom
            rec = {
                "project": tgt["project"] if in_flakebench else owner_repo.replace("/", "_"),
                "owner_repo": owner_repo,
                "test_name": key,
                "path": path,
                "sha": sha,
                "commit_subject": meta.get("subject", "")[:200],
                "category": tgt["label"] if in_flakebench else "unknown",
                "in_flakebench": in_flakebench,
                "flakebench_id": int(tgt["id"]) if in_flakebench else None,
                "code_pre": pre,
                "code_post": post,
                "n_tokens_pre": len(ta),
                "n_tokens_post": len(tb),
                "token_edit_distance": dist,
                "token_edit_distance_approx": approx,
                "diff_frac": frac,
                "annotation_only": annotation_only_change(pre, post),
                "diff": "\n".join(difflib.unified_diff(
                    pre.splitlines(), post.splitlines(), "pre", "post", lineterm="", n=2))[:4000],
                **cue_delta(pre, post, cues),
            }
            if frac > MAX_DIFF_FRAC:
                rec_ex = {k: rec[k] for k in ("project", "test_name", "sha", "diff_frac",
                                              "n_tokens_pre", "n_tokens_post", "in_flakebench")}
                rec_ex["reason"] = "diff_exceeds_40pct_of_tokens"
                out["excluded"].append(rec_ex)
                continue
            out["pairs"].append(rec)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--sha-csv", default=None)
    ap.add_argument("--max-commits", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    ctx = ctx_from_args(args)
    log = get_logger(STAGE, ctx)
    set_seed(ctx.seed)

    df = pd.read_csv(ctx.data / "prepped.csv", keep_default_na=False, na_values=[])
    cues = set(load_json(ctx.data / "cue_vocab.json")["static_union"])

    sha_csv = Path(args.sha_csv) if args.sha_csv else ctx.dataset / "filtered_tests_with_owner_sha.csv"
    commits: list[tuple[str, str]] = []
    with open(sha_csv, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            owner_repo = row[0].strip()
            for sha in dict.fromkeys(s.strip() for s in row[1].split(";") if s.strip()):
                commits.append((owner_repo, sha))
    log.info("%d unique (repo, sha) fix commits across %d repos",
             len(commits), len({o for o, _ in commits}))

    # FlakeBench flaky tests, keyed the way extract_methods keys a file
    targets: dict[str, dict] = {}
    flaky = df[df.flaky == 1]
    for r in flaky.itertuples():
        targets[r.test_name] = {"project": r.project, "label": r.label, "id": r.id}
        targets[r.test_name.split(".")[-1]] = {"project": r.project, "label": r.label, "id": r.id}
    projects_with_flaky = set(flaky.project)
    commits = [(o, s) for o, s in commits if o.replace("/", "_") in projects_with_flaky]
    log.info("%d commits in projects that have flaky tests in prepped.csv", len(commits))
    if args.max_commits:
        commits = commits[:args.max_commits]
        log.info("--max-commits: keeping %d", len(commits))

    cp = JsonlCheckpoint(ctx.ckpt / "05_mine.jsonl")
    if args.force:
        cp.reset()
    todo = [c for c in commits if f"{c[0]}@{c[1]}" not in cp]
    if len(todo) < len(commits):
        log.info("resuming: %d/%d commits already mined", len(commits) - len(todo), len(commits))

    with cp, Timer(log, f"mine {len(todo)} commits"):
        for owner_repo, sha in progress(todo, desc="commits", total=len(todo)):
            res = mine_commit(ctx.cache, owner_repo, sha, targets, cues, log)
            if res["error"]:
                log.warning("%s@%s: %s", owner_repo, sha[:8], res["error"])
            cp.put(f"{owner_repo}@{sha}", res)

    # ------------------------------------------------------------- assemble
    pairs, excluded, errors = [], [], []
    for key, res in cp.items():
        pairs.extend(res.get("pairs", []))
        excluded.extend(res.get("excluded", []))
        if res.get("error"):
            errors.append({"commit": key, "error": res["error"]})

    # de-duplicate: the same method can be touched by several fix commits
    seen, uniq = set(), []
    for p in sorted(pairs, key=lambda r: (r["project"], r["test_name"], r["sha"])):
        k = (p["project"], p["test_name"], p["code_pre"], p["code_post"])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(p)
    log.info("%d raw pairs -> %d unique; %d excluded; %d commit fetch failures",
             len(pairs), len(uniq), len(excluded), len(errors))

    cols = ["project", "owner_repo", "test_name", "path", "sha", "commit_subject", "category",
            "in_flakebench", "flakebench_id", "n_tokens_pre", "n_tokens_post",
            "token_edit_distance", "token_edit_distance_approx", "diff_frac",
            "annotation_only", "cue_stratum", "n_cue_added", "n_cue_removed",
            "cue_tokens_added", "cue_tokens_removed", "code_pre", "code_post", "diff"]
    out_df = pd.DataFrame(uniq, columns=cols) if uniq else pd.DataFrame(columns=cols)
    out_df.to_csv(ctx.data / "minimal_pairs.csv", index=False)
    pd.DataFrame(excluded).to_csv(ctx.data / "minimal_pairs_excluded.csv", index=False)

    n_fb = int(out_df.in_flakebench.sum()) if len(out_df) else 0
    res = {
        "n_commits_considered": len(commits),
        "n_commits_fetched": len(cp),
        "n_commit_errors": len(errors),
        "commit_errors": errors[:50],
        "n_pairs_raw": len(pairs),
        "n_pairs_unique": len(uniq),
        "n_pairs_flakebench_matched": n_fb,
        "n_pairs_other_test_methods": len(uniq) - n_fb,
        "n_excluded": len(excluded),
        "excluded_reasons": dict(Counter(e["reason"] for e in excluded)),
        "n_projects": int(out_df.project.nunique()) if len(out_df) else 0,
        "cue_strata": dict(Counter(out_df.cue_stratum)) if len(out_df) else {},
        "annotation_only_count": int(out_df.annotation_only.sum()) if len(out_df) else 0,
        "category_counts": dict(Counter(out_df.category)) if len(out_df) else {},
        "diff_frac": {
            "mean": float(out_df.diff_frac.mean()) if len(out_df) else None,
            "median": float(out_df.diff_frac.median()) if len(out_df) else None,
        },
        "max_diff_frac_filter": MAX_DIFF_FRAC,
        "power_note": ("Below ~40 usable pairs the binomial test has too little power and "
                       "S3 is marked inconclusive rather than passed or failed."),
        "acceptance": {
            "cue_deltas_recorded": True,
            "excluded_logged_with_reasons": True,
            "reaches_power_floor_40": len(uniq) >= 40,
            "reaches_target_of_60": len(uniq) >= 60,
        },
        "yield_vs_target": {
            "n_usable_pairs": len(uniq),
            "plan_target_band": [60, 120],
            "within_band": 60 <= len(uniq) <= 120,
            "note": ("The plan targets 60-120 usable pairs. Landing above the band is a "
                     "surplus, not a failure -- the binomial test only gains power. The "
                     "acceptance gate is the power floor of 40."),
        },
    }
    save_json(ctx.out / "05_mine.json", res)
    StageMarker(ctx, STAGE).write(f"{len(commits)}:{ctx.seed}")

    log.info("pairs: %d total (%d FlakeBench-matched, %d other test methods), %d projects",
             len(uniq), n_fb, len(uniq) - n_fb, res["n_projects"])
    log.info("cue strata: %s; annotation-only: %d", res["cue_strata"],
             res["annotation_only_count"])
    log.info("exclusions: %s", res["excluded_reasons"])
    if len(uniq) < 40:
        log.warning("only %d usable pairs -- S3 will be marked INCONCLUSIVE", len(uniq))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
