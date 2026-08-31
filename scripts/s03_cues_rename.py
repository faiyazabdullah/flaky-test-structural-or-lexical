#!/usr/bin/env python
"""03 (part 1) -- cue vocabulary and counterfactual renaming.

Produces the two artefacts the rest of the study depends on:

``data/cue_vocab.json``   the cue vocabulary, tags kept separate so each can be
                          ablated alone.  Feeds intervention A (removal) in 02.
``code_renamed``          intervention B: the source rewritten with each cue
                          replaced by a neutral identifier of matched shape.

Only the *label-independent* tags (attributed, curated, api) drive the
renaming.  Mined cues correlate with the label by construction, so building the
rewritten input from them would leak the label into the representation; they
are used only for the feature-space ablation in 02, and there they are re-mined
inside each training fold.

The transfer measurement itself (train on ``code``, evaluate on
``code_renamed``) lives in ``s02_baselines.py`` and ``s06_probe.py``, where the
models are.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from flaky.common import (StageMarker, Timer, add_common_args, ctx_from_args, file_digest,
                          get_logger, progress, save_json, set_seed)
from flaky.cues import build_rename_map, build_static_vocab, mine_cues
from flaky.javalex import code_tokens, identifiers
from flaky.rename import invert_map, rename_code, rename_stats

STAGE = "03_cues_rename"

VOCAB_NOTES = {
    "attributed": ("Tokens the plan cites from Rahman et al. (OOPSLA'25) as top "
                   "attributions. Only these carry that provenance."),
    "attributed_curated": ("Per-category extension curated in this repository. NOT from "
                           "the paper -- do not report it as such."),
    "api": "Nondeterminism-related API and type names, from the plan's list.",
    "mined_audit_only": ("Top identifiers by mutual information with the label on the FULL "
                         "dataset. Audit artefact only: never used to fit anything. The "
                         "mined cues that feed bow_ablated are re-mined inside each "
                         "training fold by s02_baselines.py."),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--mined-top-k", type=int, default=200)
    args = ap.parse_args()

    ctx = ctx_from_args(args)
    log = get_logger(STAGE, ctx)
    set_seed(ctx.seed)

    prepped = ctx.data / "prepped.csv"
    if not prepped.exists():
        log.error("missing %s -- run scripts/s01_prep.py first", prepped)
        return 1

    marker = StageMarker(ctx, STAGE)
    sig = f"{file_digest(prepped)}:{ctx.seed}:{args.mined_top_k}"
    if marker.matches(sig) and not args.force:
        log.info("already done (signature %s); use --force to redo", sig)
        return 0

    df = pd.read_csv(prepped, keep_default_na=False, na_values=[])
    log.info("loaded %d rows", len(df))
    codes = df["code"].tolist()

    # -- corpus identifiers -------------------------------------------------
    with Timer(log, "collect corpus identifiers"):
        counter: Counter = Counter()
        for c in progress(codes, desc="lex"):
            counter.update(identifiers(c))
    corpus_idents = set(counter)
    log.info("%d distinct identifiers in the corpus", len(corpus_idents))

    # -- vocabulary ---------------------------------------------------------
    tags = build_static_vocab(corpus_idents)
    with Timer(log, "mine label-correlated identifiers (audit only)"):
        mined = mine_cues(codes, df["flaky"].tolist(), top_k=args.mined_top_k)
    tags["mined_audit_only"] = mined

    static_cues = sorted(set(tags["attributed"]) | set(tags["attributed_curated"])
                         | set(tags["api"]))
    present = [c for c in static_cues if c in corpus_idents]
    log.info("cue vocabulary: attributed=%d curated=%d api=%d mined(audit)=%d; "
             "%d/%d static cues occur in the corpus",
             len(tags["attributed"]), len(tags["attributed_curated"]), len(tags["api"]),
             len(mined), len(present), len(static_cues))

    save_json(ctx.data / "cue_vocab.json", {
        "tags": tags,
        "notes": VOCAB_NOTES,
        "static_union": static_cues,
        "static_union_present_in_corpus": present,
        "renaming_uses_tags": ["attributed", "attributed_curated", "api"],
        "mined_top_k": args.mined_top_k,
    })

    # -- rename map ---------------------------------------------------------
    rename_map = build_rename_map(static_cues, corpus_idents)
    assert len(set(rename_map.values())) == len(rename_map), "rename map not injective"
    collisions = set(rename_map.values()) & corpus_idents
    assert not collisions, f"replacement collides with an existing identifier: {sorted(collisions)[:5]}"
    invert_map(rename_map)  # raises if not a bijection
    log.info("rename map: %d cues -> %d neutral identifiers, injective, no corpus collision",
             len(rename_map), len(set(rename_map.values())))
    save_json(ctx.data / "rename_map.json", rename_map)

    # -- apply --------------------------------------------------------------
    with Timer(log, "rewrite sources (intervention B)"):
        renamed = [rename_code(c, rename_map) for c in progress(codes, desc="rename")]
    df["code_renamed"] = renamed

    # -- shape-preservation evidence ----------------------------------------
    stats = [rename_stats(a, b) for a, b in zip(codes, renamed)]
    tok_equal = float(np.mean([s["java_tokens_equal"] for s in stats]))
    kinds_equal = float(np.mean([s["kinds_equal"] for s in stats]))
    n_replaced = np.asarray([s["n_replaced"] for s in stats])
    n_changed_rows = int((n_replaced > 0).sum())

    with Timer(log, "CodeBERT subword lengths for the renamed variant"):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("microsoft/codebert-base",
                                            cache_dir=str(ctx.cache / "hf"))
        sub_after = []
        for i in progress(range(0, len(renamed), 256), desc="tokenize",
                          total=(len(renamed) + 255) // 256):
            enc = tok(renamed[i:i + 256], add_special_tokens=True, truncation=False,
                      padding=False)["input_ids"]
            sub_after.extend(len(e) for e in enc)
    sub_after = np.asarray(sub_after)
    sub_before = df["n_tokens"].to_numpy()
    sub_delta = sub_after - sub_before

    res = {
        "n_rows": int(len(df)),
        "n_cues_static": len(static_cues),
        "n_cues_present": len(present),
        "rename_map_size": len(rename_map),
        "rows_with_at_least_one_replacement": n_changed_rows,
        "rows_with_replacement_rate": float(n_changed_rows / len(df)),
        "mean_replacements_per_row": float(n_replaced.mean()),
        "java_token_count_equal_rate": tok_equal,
        "java_token_kinds_equal_rate": kinds_equal,
        "codebert_subword_delta": {
            "mean": float(sub_delta.mean()),
            "median": float(np.median(sub_delta)),
            "equal_rate": float((sub_delta == 0).mean()),
            "p05": float(np.percentile(sub_delta, 5)),
            "p95": float(np.percentile(sub_delta, 95)),
            "note": ("Renaming is exactly token-count preserving at the Java lexical level "
                     "(that is the acceptance criterion). Subword counts still shift because "
                     "a neutral name segments differently; reported here rather than hidden."),
        },
        "cue_tag_sizes": {k: len(v) for k, v in tags.items()},
        "top_replacements": sorted(
            ((counter.get(k, 0), k, v) for k, v in rename_map.items()), reverse=True)[:25],
    }
    res["acceptance"] = {
        "rename_unit_tests": "see tests/test_rename.py -- run via scripts/run_all.py --tests",
        "java_token_count_equal_ge_95pct": tok_equal >= 0.95,
        "rename_map_injective": True,
        "no_collision_with_corpus_identifiers": True,
    }

    df.to_csv(prepped, index=False)
    save_json(ctx.out / "03_rename.json", res)
    marker.write(sig, {"n_cues": len(rename_map)})

    log.info("java token count preserved on %.4f of rows (acceptance >= 0.95)", tok_equal)
    log.info("rows touched by renaming: %d (%.1f%%), mean %.2f replacements",
             n_changed_rows, 100 * n_changed_rows / len(df), n_replaced.mean())
    log.info("CodeBERT subword delta: mean %+.2f, unchanged on %.3f of rows",
             sub_delta.mean(), (sub_delta == 0).mean())
    ok = tok_equal >= 0.95
    log.info("acceptance java_token_count_equal_ge_95pct %s", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
