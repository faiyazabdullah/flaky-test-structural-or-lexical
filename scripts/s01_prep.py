#!/usr/bin/env python
"""01 -- data preparation.

Builds ``data/prepped.csv`` from the FlakeBench CSV.  Everything downstream
assumes this file's row order.

The formatting normalisation here is mandatory, not a nicety: in the released
CSV the two classes are separable by whitespace alone (four booleans give AUROC
1.000), because the classes were extracted by different pipelines.  A subword
tokenizer encodes leading whitespace, so skipping this step means every
downstream result measures provenance.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from flaky.common import (Ctx, StageMarker, Timer, add_common_args, ctx_from_args,
                          file_digest, get_logger, progress, report_acceptance, save_json,
                          set_seed)
from flaky.normalize import WHITESPACE_FEATURES, normalize_code, whitespace_features

STAGE = "01_prep"
CATEGORY_SLUG = {
    "non-flaky": "non_flaky",
    "async wait": "async_wait",
    "concurrency": "concurrency",
    "time": "time",
    "unordered collections": "unordered_collections",
    "test order dependency": "test_order_dependency",
}


def whitespace_auroc(codes, y) -> float:
    """AUROC of a logistic regression on the four whitespace booleans.

    Constant features (the post-normalisation case) give a constant score and
    therefore exactly 0.5, which is the number the acceptance check wants.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    X = np.asarray([[whitespace_features(c)[k] for k in WHITESPACE_FEATURES] for c in codes],
                   dtype=np.float64)
    y = np.asarray(y)
    if X.std(axis=0).max() == 0.0:
        return 0.5
    clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X, y)
    return float(roc_auc_score(y, clf.decision_function(X)))


def smoke_subset(df: pd.DataFrame, seed: int, n_projects: int = 12,
                 flaky_per_project: int = 12, nonflaky_per_project: int = 45) -> pd.DataFrame:
    """A small but structurally faithful slice: enough projects for grouped CV,
    every root-cause category represented, both classes present."""
    rng = np.random.RandomState(seed)
    flaky_counts = df[df.flaky == 1].groupby("project").size().sort_values(ascending=False)
    projects = list(flaky_counts.index[:n_projects])
    keep = []
    for p in projects:
        sub = df[df.project == p]
        f = sub[sub.flaky == 1]
        nf = sub[sub.flaky == 0]
        keep.append(f.head(flaky_per_project))
        if len(nf):
            take = min(nonflaky_per_project, len(nf))
            keep.append(nf.iloc[rng.choice(len(nf), size=take, replace=False)])
    out = pd.concat(keep).sort_values("id").reset_index(drop=True)
    # make sure every category survives the subsetting
    missing = set(df.label.unique()) - set(out.label.unique())
    for lab in sorted(missing):
        out = pd.concat([out, df[df.label == lab].head(3)])
    return out.drop_duplicates("id").sort_values("id").reset_index(drop=True)


def match_on_length(df: pd.DataFrame, caliper: float = 0.05, seed: int = 0) -> np.ndarray:
    """1:1 nearest-neighbour match on ``n_tokens``, relative caliper, preferring
    a partner within the same project and falling back to global.

    Solved as a minimum-cost assignment rather than greedily.  The cost ladder
    encodes the plan's preference order exactly -- feasibility (inside the
    caliper) dominates same-project, which dominates closeness in length -- and
    a greedy pass over the same rule leaves the class medians further apart
    than they need to be, which is the thing the matched subset exists to fix.
    """
    from scipy.optimize import linear_sum_assignment

    flaky = df.index[df.flaky == 1].to_numpy()
    pool = df.index[df.flaky == 0].to_numpy()
    matched = np.zeros(len(df), dtype=bool)
    if len(flaky) == 0 or len(pool) == 0:
        return matched

    tok = df["n_tokens"].to_numpy().astype(np.float64)
    proj = df["project"].to_numpy()

    INFEASIBLE = 1e9
    CROSS_PROJECT = 1e4      # > any in-caliper length gap, < INFEASIBLE

    d = np.abs(tok[pool][None, :] - tok[flaky][:, None])
    tol = np.maximum(1.0, caliper * tok[flaky])[:, None]
    cost = d.copy()
    cost += (proj[pool][None, :] != proj[flaky][:, None]) * CROSS_PROJECT
    cost[d > tol] = INFEASIBLE

    rows, cols = linear_sum_assignment(cost)
    for r, c in zip(rows, cols):
        if cost[r, c] >= INFEASIBLE:
            continue
        matched[df.index.get_loc(flaky[r])] = True
        matched[df.index.get_loc(pool[c])] = True
    return matched


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--input", default=None,
                    help="FlakeBench CSV (default dataset/FlakeBench_dataset.csv)")
    ap.add_argument("--on-post-norm-conflict", choices=("drop", "stop"), default="drop",
                    help=("what to do when two rows normalise to identical code but carry "
                          "different labels: drop both copies (default) or stop the run"))
    args = ap.parse_args()

    ctx = ctx_from_args(args)
    log = get_logger(STAGE, ctx)
    set_seed(ctx.seed)

    src = Path(args.input) if args.input else ctx.dataset / "FlakeBench_dataset.csv"
    marker = StageMarker(ctx, STAGE)
    sig = f"{file_digest(src)}:{ctx.seed}:{int(ctx.smoke)}:{args.on_post_norm_conflict}"
    if marker.matches(sig) and not args.force and (ctx.data / "prepped.csv").exists():
        log.info("already done (signature %s); use --force to redo", sig)
        return 0

    res: dict = {"input": str(src), "smoke": ctx.smoke, "seed": ctx.seed}

    with Timer(log, "read FlakeBench CSV"):
        raw = pd.read_csv(src, dtype={"id": "int64"}, keep_default_na=False, na_values=[])
    log.info("raw rows=%d cols=%s", len(raw), list(raw.columns))
    res["n_rows_raw"] = int(len(raw))
    res["class_counts_raw"] = raw["label"].value_counts().to_dict()

    # -- conflicting labels: stop and report rather than picking a winner ----
    conflicts = (raw.groupby("full_code")["label"].nunique() > 1)
    n_conf = int(conflicts.sum())
    res["n_conflicting_label_codes"] = n_conf
    if n_conf:
        bad = conflicts[conflicts].index[:5]
        log.error("%d code strings carry conflicting labels, e.g. %r", n_conf, list(bad)[:1])
        raise SystemExit("conflicting labels present -- refusing to pick a winner")
    log.info("no code string carries conflicting labels")

    # -- dedup on exact full_code ------------------------------------------
    before = len(raw)
    df = raw.drop_duplicates(subset="full_code", keep="first").reset_index(drop=True)
    res["n_dup_exact_removed"] = int(before - len(df))
    log.info("dedup on raw full_code: %d -> %d (-%d)", before, len(df), before - len(df))

    # -- the artefact, measured before we remove it -------------------------
    y_raw = (df["label"] != "non-flaky").astype(int).to_numpy()
    res["whitespace_auroc_before"] = whitespace_auroc(df["full_code"].tolist(), y_raw)
    res["whitespace_rates_before"] = {
        k: float(np.mean([whitespace_features(c)[k] for c in df["full_code"]]))
        for k in WHITESPACE_FEATURES}
    res["whitespace_rates_before_by_class"] = {
        cls: {k: float(np.mean([whitespace_features(c)[k]
                                for c in df.loc[y_raw == v, "full_code"]]))
              for k in WHITESPACE_FEATURES}
        for cls, v in (("non_flaky", 0), ("flaky", 1))}
    log.info("whitespace-only AUROC BEFORE normalisation: %.4f",
             res["whitespace_auroc_before"])

    # -- normalise ----------------------------------------------------------
    with Timer(log, "normalise formatting"):
        df["code"] = [normalize_code(c) for c in progress(df["full_code"].tolist(),
                                                          desc="normalise")]

    # Normalisation exposes a second, distinct contradiction: rows whose code
    # differs only in whitespace but whose labels disagree.  In this release
    # they are duplicated wildfly tests carried once as `test order dependency`
    # (under a SHA-prefixed synthetic test name) and once as `non-flaky` (under
    # the real class name).  Neither label can be trusted, so both copies go --
    # keeping either would be picking a winner.  The rows are recorded in full.
    conf2 = (df.groupby("code")["label"].nunique() > 1)
    conflicted_codes = set(conf2[conf2].index)
    res["n_conflicting_label_codes_post_norm"] = len(conflicted_codes)
    if conflicted_codes:
        rows = df[df.code.isin(conflicted_codes)]
        res["post_norm_conflicts"] = {
            "policy": args.on_post_norm_conflict,
            "n_codes": len(conflicted_codes),
            "n_rows": int(len(rows)),
            "rows": rows[["id", "project", "test_name", "label"]].to_dict("records"),
            "note": ("Identical code carrying contradictory labels. Not resolvable from the "
                     "data; both copies are dropped so neither label is silently preferred."),
        }
        log.warning("%d normalised code strings carry conflicting labels (%d rows): %s",
                    len(conflicted_codes), len(rows), args.on_post_norm_conflict)
        for r in rows.itertuples():
            log.warning("  conflict id=%s project=%s test=%s label=%r",
                        r.id, r.project, r.test_name, r.label)
        if args.on_post_norm_conflict == "stop":
            raise SystemExit("conflicting labels after normalisation -- refusing to pick a winner")
        df = df[~df.code.isin(conflicted_codes)]
        res["n_post_norm_conflict_rows_dropped"] = int(len(rows))
    else:
        res["n_post_norm_conflict_rows_dropped"] = 0

    before = len(df)
    df = df.drop_duplicates(subset="code", keep="first").reset_index(drop=True)
    res["n_dup_post_norm_removed"] = int(before - len(df))
    log.info("dedup on normalised code: %d -> %d (-%d)", before, len(df), before - len(df))

    df = df[df["code"].str.len() > 0].reset_index(drop=True)
    res["n_rows_after_dedup"] = int(len(df))

    # -- post-normalisation assertions --------------------------------------
    lead = float(np.mean([c[:1] in (" ", "\t") for c in df["code"]]))
    ends_nl = float(np.mean([c.endswith("\n") for c in df["code"]]))
    res["post_norm_leading_indent_rate"] = lead
    res["post_norm_ends_with_newline_rate"] = ends_nl
    assert lead == 0.0, f"leading-indent rate is {lead}, must be exactly 0"
    assert ends_nl == 0.0, f"{ends_nl:.4f} of rows still end in a newline"
    log.info("post-normalisation: leading-indent rate %.1f, ends-with-newline rate %.1f",
             lead, ends_nl)

    y = (df["label"] != "non-flaky").astype(int).to_numpy()
    res["whitespace_auroc_after"] = whitespace_auroc(df["code"].tolist(), y)
    log.info("whitespace-only AUROC AFTER normalisation: %.4f (acceptance: 0.50 +- 0.05)",
             res["whitespace_auroc_after"])
    assert abs(res["whitespace_auroc_after"] - 0.5) <= 0.05, "normalisation incomplete"

    # -- derived columns ----------------------------------------------------
    # FlakeBench's `id` is NOT unique: the flaky and non-flaky halves were
    # numbered independently and 66 ids collide across them. Anything keyed on
    # `id` -- a checkpoint, a join -- silently mixes two different tests, so a
    # content-derived key is added and used everywhere downstream. `id` is kept
    # for provenance.
    import hashlib
    df["uid"] = [hashlib.sha1(c.encode("utf-8")).hexdigest()[:12] for c in df["code"]]
    assert df["uid"].is_unique, "uid collision -- code was not deduplicated"
    res["n_colliding_source_ids"] = int(df["id"].duplicated().sum())
    if res["n_colliding_source_ids"]:
        log.warning("%d rows share an `id` with another row (FlakeBench numbers its flaky "
                    "and non-flaky halves independently); using content-derived `uid` as the "
                    "join and checkpoint key throughout", res["n_colliding_source_ids"])

    df["flaky"] = y
    df["category_name"] = df["label"].map(CATEGORY_SLUG).fillna("other")
    df["n_chars"] = df["code"].str.len().astype("int64")
    df["n_lines"] = df["code"].str.count("\n").add(1).astype("int64")

    if ctx.smoke:
        before = len(df)
        df = smoke_subset(df, ctx.seed)
        log.info("SMOKE subset: %d -> %d rows, %d projects, %d flaky",
                 before, len(df), df.project.nunique(), int(df.flaky.sum()))

    with Timer(log, "CodeBERT tokenizer lengths"):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("microsoft/codebert-base",
                                            cache_dir=str(ctx.cache / "hf"))
        lens = []
        codes = df["code"].tolist()
        for i in progress(range(0, len(codes), 256), desc="tokenize",
                          total=(len(codes) + 255) // 256):
            enc = tok(codes[i:i + 256], add_special_tokens=True, truncation=False,
                      padding=False)["input_ids"]
            lens.extend(len(e) for e in enc)
        df["n_tokens"] = np.asarray(lens, dtype=np.int64)

    # -- length-matched subset ---------------------------------------------
    with Timer(log, "length-matched subset"):
        df["matched"] = match_on_length(df, caliper=0.05, seed=ctx.seed)
    m = df[df.matched]
    res["matched"] = {
        "n": int(len(m)),
        "n_flaky": int(m.flaky.sum()),
        "n_non_flaky": int((1 - m.flaky).sum()),
        "n_flaky_unmatched": int(df.flaky.sum() - m.flaky.sum()),
        "median_n_tokens_flaky": float(m.loc[m.flaky == 1, "n_tokens"].median()),
        "median_n_tokens_non_flaky": float(m.loc[m.flaky == 0, "n_tokens"].median()),
        "n_projects": int(m.project.nunique()),
    }
    a, b = res["matched"]["median_n_tokens_flaky"], res["matched"]["median_n_tokens_non_flaky"]
    res["matched"]["median_rel_diff"] = float(abs(a - b) / max(a, b)) if max(a, b) else 0.0
    log.info("matched subset: n=%d (%d/%d), median n_tokens %.0f vs %.0f (rel diff %.4f)",
             res["matched"]["n"], res["matched"]["n_flaky"], res["matched"]["n_non_flaky"],
             a, b, res["matched"]["median_rel_diff"])

    # -- length signal on the full set, for the record ----------------------
    from sklearn.metrics import roc_auc_score
    res["length_auroc_full"] = float(roc_auc_score(df.flaky, df.n_tokens))
    res["length_auroc_matched"] = float(roc_auc_score(m.flaky, m.n_tokens)) if len(m) else None
    res["median_chars_flaky"] = float(df.loc[df.flaky == 1, "n_chars"].median())
    res["median_chars_non_flaky"] = float(df.loc[df.flaky == 0, "n_chars"].median())
    log.info("length-only AUROC: full %.4f, matched %.4f",
             res["length_auroc_full"], res["length_auroc_matched"] or float("nan"))

    # -- write --------------------------------------------------------------
    cols = ["uid", "id", "project", "test_name", "code", "flaky", "label", "category",
            "category_name", "n_chars", "n_lines", "n_tokens", "matched"]
    out = df[cols].reset_index(drop=True)
    out.to_csv(ctx.data / "prepped.csv", index=False)

    res["n_rows_final"] = int(len(out))
    res["n_projects"] = int(out.project.nunique())
    res["class_counts_final"] = out["label"].value_counts().to_dict()
    res["n_flaky"] = int(out.flaky.sum())
    res["class_prior"] = float(out.flaky.mean())
    res["acceptance"] = {
        "no_duplicate_code": bool(out.code.duplicated().sum() == 0),
        "uid_unique": bool(out.uid.is_unique),
        "leading_indent_rate_zero": lead == 0.0,
        "whitespace_auroc_within_tolerance": abs(res["whitespace_auroc_after"] - 0.5) <= 0.05,
        "matched_balanced": res["matched"]["n_flaky"] == res["matched"]["n_non_flaky"],
        "matched_median_within_2pct": res["matched"]["median_rel_diff"] <= 0.02,
    }
    save_json(ctx.out / "01_prep.json", res)
    marker.write(sig, {"n_rows": res["n_rows_final"]})

    log.info("wrote %s (%d rows, %d projects, %d flaky, prior %.4f)",
             ctx.data / "prepped.csv", res["n_rows_final"], res["n_projects"],
             res["n_flaky"], res["class_prior"])
    return 0 if report_acceptance(log, res["acceptance"], ctx.smoke) else 1


if __name__ == "__main__":
    raise SystemExit(main())
