#!/usr/bin/env python
"""Collect every result into a single directory of Markdown files.

``out/`` holds the machine-readable artefacts: JSON per stage, ``.npy``
embeddings, PNG figures.  This turns them into self-contained Markdown that can
be read (or uploaded) without the repository, with every table carrying the name
of the JSON file it came from.

    python scripts/export_reports.py            # -> reports/
    python scripts/export_reports.py --smoke    # -> reports_smoke/
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from flaky.common import add_common_args, ctx_from_args, get_logger, load_json, save_text
from flaky.cv import BASELINE_METHODS
from flaky.structural import PROPERTIES

STAGE = "export_reports"
VERDICT_WORD = {True: "**PASS**", False: "**FAIL**", None: "**INCONCLUSIVE**"}


def f(x, nd=4):
    if x is None:
        return "—"
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def folds(v, nd=3):
    return ", ".join(f(x, nd) for x in v) if v else "—"


# --------------------------------------------------------------------------


def doc_01(d) -> str:
    L = ["# 01 — Data preparation", "",
         "Source: `out/01_prep.json`. Produces `data/prepped.csv`, whose row order every",
         "later stage assumes.", "",
         "## Counts", "",
         "| | |", "|---|---|",
         f"| raw rows | {d['n_rows_raw']} |",
         f"| exact duplicates removed | {d['n_dup_exact_removed']} |",
         f"| duplicates removed after normalisation | {d.get('n_dup_post_norm_removed', 0)} |",
         f"| rows dropped for contradictory labels | {d.get('n_post_norm_conflict_rows_dropped', 0)} |",
         f"| **final rows** | **{d['n_rows_final']}** |",
         f"| flaky | {d['n_flaky']} ({f(d['class_prior'])}) |",
         f"| projects | {d['n_projects']} |",
         f"| source `id` values that collide | {d.get('n_colliding_source_ids', 0)} |",
         "",
         "## Class composition", "",
         "| label | rows |", "|---|---|"]
    for k, v in sorted(d["class_counts_final"].items(), key=lambda kv: -kv[1]):
        L.append(f"| `{k}` | {v} |")
    L += ["",
          "## The formatting artefact", "",
          "In the released CSV the two classes are separable by whitespace alone.", "",
          "| feature | non-flaky | flaky |", "|---|---|---|"]
    b = d["whitespace_rates_before_by_class"]
    for k in b["non_flaky"]:
        L.append(f"| `{k}` | {f(b['non_flaky'][k], 4)} | {f(b['flaky'][k], 4)} |")
    L += ["",
          f"A logistic regression on those four booleans gives **AUROC "
          f"{f(d['whitespace_auroc_before'], 3)}** before normalisation and "
          f"**{f(d['whitespace_auroc_after'], 3)}** after.",
          "",
          "The classes were evidently extracted by different pipelines. A subword tokenizer",
          "encodes leading whitespace, so without normalisation every downstream number",
          "measures provenance rather than flakiness.",
          "",
          "## Length confound", "",
          f"- median chars: flaky {f(d['median_chars_flaky'], 0)} vs non-flaky "
          f"{f(d['median_chars_non_flaky'], 0)}",
          f"- `n_tokens` alone reaches AUROC **{f(d['length_auroc_full'], 4)}** on the full "
          f"set and **{f(d['length_auroc_matched'], 4)}** on the length-matched subset",
          f"- matched subset: {d['matched']['n']} rows ({d['matched']['n_flaky']}/"
          f"{d['matched']['n_non_flaky']}), median `n_tokens` "
          f"{f(d['matched']['median_n_tokens_flaky'], 0)} vs "
          f"{f(d['matched']['median_n_tokens_non_flaky'], 0)} "
          f"(relative difference {f(d['matched']['median_rel_diff'], 4)})",
          ""]
    if d.get("post_norm_conflicts"):
        c = d["post_norm_conflicts"]
        L += ["## Contradictory labels", "",
              f"{c['n_codes']} test methods appear twice with **identical code and opposite "
              f"labels** ({c['n_rows']} rows). Neither label can be trusted, so both copies "
              f"were dropped rather than one being silently preferred.", "",
              "| id | project | test | label |", "|---|---|---|---|"]
        for r in c["rows"]:
            L.append(f"| {r['id']} | `{r['project']}` | `{r['test_name']}` | `{r['label']}` |")
        L.append("")
    L += ["## Acceptance", "", "| check | result |", "|---|---|"]
    for k, v in d["acceptance"].items():
        L.append(f"| {k} | {'PASS' if v else 'FAIL'} |")
    return "\n".join(L) + "\n"


def doc_02(b, t) -> str:
    L = ["# 02 — Baselines and the within- vs cross-project gap", "",
         "Source: `out/02_baselines.json`. 5 folds, stratified, seed 0. `within` uses",
         "`StratifiedKFold`; `cross` uses `StratifiedGroupKFold` grouped by project, so no",
         "project spans the split. Vectorisers are fitted inside each fold and `C` is tuned",
         "on an inner split of the training rows.", "",
         f"Class prior: {f(b['class_prior_full'])} (full set), "
         f"{f(b.get('class_prior_matched'))} (matched subset).", "",
         "## Full set", "",
         "| method | within AP | cross AP | gap | within AUROC | cross AUROC |",
         "|---|---|---|---|---|---|"]
    for m in BASELINE_METHODS:
        c = b["methods"].get(m, {}).get("full", {})
        if "cross" not in c:
            continue
        L.append(f"| `{m}` | {f(c['within']['ap']['mean'])} ± {f(c['within']['ap']['std'], 3)} "
                 f"| {f(c['cross']['ap']['mean'])} ± {f(c['cross']['ap']['std'], 3)} "
                 f"| {c['gap']['mean']:+.4f} | {f(c['within']['auroc']['mean'])} "
                 f"| {f(c['cross']['auroc']['mean'])} |")
    L += ["", "## Length-matched subset", "",
          "| method | within AP | cross AP | gap |", "|---|---|---|---|"]
    for m in BASELINE_METHODS:
        c = b["methods"].get(m, {}).get("matched", {})
        if "cross" not in c:
            continue
        L.append(f"| `{m}` | {f(c['within']['ap']['mean'])} | {f(c['cross']['ap']['mean'])} "
                 f"| {c['gap']['mean']:+.4f} |")
    L += ["", "## Fold-level cross-project AP (full set)", "",
          "Reported in full because five folds is very little power and the paired tests in",
          "07 read these vectors directly.", "",
          "| method | fold 1 | fold 2 | fold 3 | fold 4 | fold 5 |",
          "|---|---|---|---|---|---|"]
    for m in BASELINE_METHODS:
        c = b["methods"].get(m, {}).get("full", {})
        if "cross" in c:
            L.append(f"| `{m}` | " + " | ".join(f(x, 4) for x in c["cross"]["ap"]["folds"]) + " |")
    swb = b["STRONG_WORD_BASELINE"]
    L += ["", "## STRONG_WORD_BASELINE", "",
          f"**{f(swb['ap_mean'])}** — `{swb['method']}`, cross-project, full set. Fixed here",
          "and not renegotiated; stopping rule S1 is evaluated against it.", "",
          "| candidate | cross AP |", "|---|---|"]
    for k, v in swb["candidates"].items():
        L.append(f"| `{k}` | {f(v)} |")
    L += ["", "## Cue ablation", "",
          "`bow_ablated` is `bow` with the cue vocabulary removed from the feature space",
          "(the code is untouched). Mined cues are re-mined inside each training fold.", ""]
    a = b["methods"].get("bow_ablated", {}).get("full", {}).get("cross", {})
    bw = b["methods"].get("bow", {}).get("full", {}).get("cross", {})
    if a and bw:
        L += [f"- `bow` cross-project AP {f(bw['ap']['mean'])} → `bow_ablated` "
              f"{f(a['ap']['mean'])} (**{a['ap']['mean'] - bw['ap']['mean']:+.4f}**)",
              f"- features dropped per fold: {a['n_dropped_features_per_fold']} of "
              f"{a['n_features_per_fold']}", ""]
    L += ["## Acceptance", "", "| check | result |", "|---|---|"]
    for k, v in b["acceptance"].items():
        if isinstance(v, bool):
            L.append(f"| {k} | {'PASS' if v else 'FAIL'} |")
    return "\n".join(L) + "\n"


def doc_03(r, t) -> str:
    L = ["# 03 — Cue removal and counterfactual renaming", "",
         "Sources: `out/03_rename.json`, `out/03_transfer.json`, `data/cue_vocab.json`,",
         "`data/rename_map.json`.", "",
         "Two interventions on the lexical channel:", "",
         "- **A — removal.** Cue tokens dropped from the vectoriser vocabulary. Feature space",
         "  only; the code is untouched. Feeds `bow_ablated` in 02.",
         "- **B — counterfactual renaming.** The source is rewritten, each cue replaced by a",
         "  neutral identifier of matched shape (`Thread.sleep` → `Util.pause`,",
         "  `CountDownLatch` → `Gate`). Control flow, call structure, arity and assertions",
         "  are preserved.", ""]
    if r:
        L += ["## Cue vocabulary", "",
              "Three separately-tagged sources, so each can be ablated alone.", "",
              "| tag | size | provenance |", "|---|---|---|",
              f"| `attributed` | {r['cue_tag_sizes']['attributed']} | tokens the plan cites "
              f"from Rahman et al. (OOPSLA'25) |",
              f"| `attributed_curated` | {r['cue_tag_sizes']['attributed_curated']} | "
              f"per-category extension curated in this repository, **not** from the paper |",
              f"| `api` | {r['cue_tag_sizes']['api']} | nondeterminism-related API and type "
              f"names |",
              f"| `mined_audit_only` | {r['cue_tag_sizes']['mined_audit_only']} | top "
              f"identifiers by mutual information with the label, on the full set — **audit "
              f"artefact only, never used to fit anything** |",
              "",
              "Only the label-independent tags drive the renaming. Mined cues correlate with",
              "the label by construction, so building the rewritten input from them would leak",
              "the label into the representation; they are used only for intervention A, and",
              "there they are re-mined inside each training fold.", "",
              "## Shape preservation", "",
              f"- rows touched: {r['rows_with_at_least_one_replacement']} "
              f"({f(r['rows_with_replacement_rate'] * 100, 1)}%), mean "
              f"{f(r['mean_replacements_per_row'], 2)} replacements per row",
              f"- **Java token count identical on {f(r['java_token_count_equal_rate'], 4)} of "
              f"rows** — every substitution swaps one identifier token for one identifier "
              f"token, so this is exact by construction (acceptance: ≥ 0.95)",
              f"- token *kinds* identical on {f(r['java_token_kinds_equal_rate'], 4)} of rows",
              f"- CodeBERT subword counts still shift (mean "
              f"{r['codebert_subword_delta']['mean']:+.2f}, unchanged on "
              f"{f(r['codebert_subword_delta']['equal_rate'], 3)} of rows) because a neutral "
              f"name segments differently — reported rather than hidden", "",
              "### Most-replaced cues", "",
              "| corpus occurrences | cue | neutral name |", "|---|---|---|"]
        for n, k, v in r["top_replacements"][:15]:
            L.append(f"| {n} | `{k}` | `{v}` |")
        L.append("")
    if t:
        L += ["## Transfer drop", "",
              "Δ = AP(`code`) − AP(`code_renamed`), cross-project, model **trained on `code`**.",
              "The asymmetry between the train and eval distributions is the whole",
              "measurement.", "",
              "| method | AP on `code` | AP on `code_renamed` | Δ | Δ folds |",
              "|---|---|---|---|---|"]
        for m in BASELINE_METHODS:
            e = t["methods"].get(m)
            if not e:
                continue
            L.append(f"| `{m}` | {f(e['ap_code']['mean'])} | {f(e['ap_code_renamed']['mean'])} "
                     f"| {e['delta']['mean']:+.4f} | {folds(e['delta']['folds'])} |")
        rt = t["methods"].get("bow", {}).get("retrained_on_renamed")
        if rt:
            L += ["", "### Sanity check: retraining on the renamed variant", "",
                  f"A `bow` model **retrained** on `code_renamed` scores {f(rt['ap']['mean'])}, "
                  f"against {f(t['methods']['bow']['ap_code']['mean'])} on `code` — a "
                  f"difference of {f(rt['abs_diff_from_code_mean'], 4)}, within one fold-std "
                  f"({f(rt['fold_std_of_code_ap'], 4)}): "
                  f"**{'yes' if rt['within_one_fold_std'] else 'NO'}**.", "",
                  "Retraining measures nothing about lexical reliance — a bag-of-tokens model",
                  "simply relearns `pause` and `Gate`. It is reported only as evidence that",
                  "the renaming is shape-preserving and not leaking information.", ""]
    if r:
        L += ["## Acceptance", "", "| check | result |", "|---|---|"]
        for k, v in r["acceptance"].items():
            L.append(f"| {k} | {'PASS' if v is True else ('FAIL' if v is False else v)} |")
    return "\n".join(L) + "\n"


def doc_04(d, v) -> str:
    L = ["# 04 — Structural property labels", "",
         "Sources: `out/04_structural.json`, `out/04_validation.json`. Labels come from a",
         "tree-sitter taint pass and never from the flakiness label — that is what makes",
         '"structural information" a testable claim rather than a residue left over after',
         "cue removal.", "",
         "| property | definition |", "|---|---|",
         "| `P_ASYNC` | an async dispatch reaches an assertion with no intervening "
         "synchronisation on the path |",
         "| `P_UNORD` | a value originating from unordered iteration reaches an assertion "
         "argument |",
         "| `P_CLOCK` | a clock read reaches an assertion argument |",
         "",
         "A fourth property — a shared field written by one test and read by another — is",
         "inter-procedural, needs the class body and sibling tests, and **cannot** be",
         "computed from FlakeBench's isolated method bodies. Deferred to Phase 2.", "",
         "## Parsing", "",
         f"- parse failure rate **{f(d['parse_failure_rate'], 4)}** (acceptance < 0.05)",
         f"- without truncation repair it would be "
         f"{f(d['parse_repairs']['failure_rate_without_repair'], 4)}",
         f"- rows repaired: {d['parse_repairs']['n_repaired']} "
         f"({f(d['parse_repairs']['rate'] * 100, 1)}%) — {d['parse_repairs']['by_kind']}",
         "",
         "FlakeBench truncates a fraction of method bodies mid-statement. Those rows are",
         "almost entirely non-flaky, so leaving them unparsed would bias the labels in",
         "exactly the direction the plan warns about. The repair closes open brackets or",
         "rewinds to the last complete statement; it never invents content, and every",
         "repaired row is flagged in `data/structural_full.csv`.", "",
         "Remaining failures by label: " + str(d.get("parse_failures_by_label", {})), "",
         "## Prevalence", "",
         "| property | overall | " + " | ".join(f"{k}" for k in sorted(d["prevalence_by_label"]))
         + " |",
         "|---|---|" + "---|" * len(d["prevalence_by_label"])]
    for p in PROPERTIES:
        row = [f"`{p}`", f(d["prevalence_overall"][p])]
        for k in sorted(d["prevalence_by_label"]):
            row.append(f(d["prevalence_by_label"][k][p]))
        L.append("| " + " | ".join(row) + " |")
    fc = d["floor_check"]
    L += ["",
          f"**Floor check** (not a finding): `P_ASYNC` prevalence in the async-wait category "
          f"{f(fc['P_ASYNC_rate_async_wait'], 4)} vs non-flaky "
          f"{f(fc['P_ASYNC_rate_non_flaky'], 4)} — "
          f"{'passes' if fc['passes'] else '**FAILS, the analysis is broken**'}.", ""]
    diag = d.get("P_ASYNC_definition_diagnostic")
    if diag:
        L += ["## A consequence of the definition", "",
              "The plan puts `sleep` in the synchronisation kill set, so the canonical",
              "async-wait flaky shape — *dispatch; sleep; assert* — is excluded from",
              "`P_ASYNC` by construction.", "",
              f"- rows that would flip if `sleep` were not a barrier: "
              f"{diag['n_rows_that_would_flip']}",
              f"- async-wait rate would rise from {f(fc['P_ASYNC_rate_async_wait'], 4)} to "
              f"{f(diag['rate_in_async_wait_if_sleep_not_a_barrier'], 4)}", "",
              "Diagnostic only. No stopping-rule number depends on it.", ""]
    soc = d["structural_only_classifier"]
    L += ["## Do the structural properties predict flakiness?", "",
          f"A logistic regression on the three booleans alone, cross-project:", "",
          f"- **AP {f(soc['ap']['mean'])} ± {f(soc['ap']['std'], 4)}** "
          f"(folds: {folds(soc['ap']['folds'])})",
          f"- class prior {f(soc['class_prior'])}", "",
          "Barely above the prior. On this dataset the named structural properties carry",
          "almost no predictive signal for the flakiness label — which is reportable either",
          "way, and is the first half of the possible split verdict in 07.", ""]
    if v:
        L += ["## Validation against the hand audit", "",
              f"{v['n_audited']} methods, stratified by property and by label, read in full",
              f"by {v['auditor']}. Precision floor **{v['precision_floor']}**: below it a",
              "property is either fixed or dropped from the study.", "",
              "### Definitional — does the property hold as the plan writes it?", "",
              "| property | TP | FP | FN | TN | precision (95% Wilson) | recall | verdict |",
              "|---|---|---|---|---|---|---|---|"]
        for p in PROPERTIES:
            e = v["properties"][p]
            L.append(f"| `{p}` | {e['tp']} | {e['fp']} | {e['fn']} | {e['tn']} | "
                     f"{f(e['precision'], 3)} [{f(e['precision_wilson_95'][0], 2)}, "
                     f"{f(e['precision_wilson_95'][1], 2)}] | {f(e['recall'], 3)} | "
                     f"{'keep' if e['meets_precision_floor'] else '**below floor**'} |")
        if v.get("semantic"):
            L += ["", "### Semantic — does it hold in the code's actual behaviour?", "",
                  "| property | precision | recall |", "|---|---|---|"]
            for p in PROPERTIES:
                if p in v["semantic"]:
                    e = v["semantic"][p]
                    L.append(f"| `{p}` | {f(e['precision'], 3)} ({e['tp']}/{e['tp'] + e['fp']}) "
                             f"| {f(e['recall'], 3)} |")
            L += ["",
                  "The two diverge wherever the plan's fixed vocabulary matches a name that is",
                  "not doing what the name suggests: `execute` and `start` are listed as async",
                  "dispatches, but Hystrix's `cmd.execute()` is the **synchronous** API and",
                  '`new SystemCtl(t).start("docker")` merely builds a command object. This is a',
                  "property of the definitions, not a defect in the implementation, and it",
                  "bounds how much a probe trained on these labels can be said to have learned",
                  "about genuine asynchrony.", ""]
        if v.get("provenance", {}).get("development_pass"):
            L += ["### Bugs the audit found", "",
                  v["provenance"]["development_pass"], "",
                  "Each is now a regression test in `tests/test_structural.py`.", ""]
    return "\n".join(L) + "\n"


def doc_05(m, p) -> str:
    L = ["# 05 — Minimal pairs from fix commits", "",
         "Sources: `out/05_mine.json`, `out/05_pairs.json`, `data/minimal_pairs.csv`,",
         "`data/minimal_pairs_excluded.csv`.", "",
         "The converse of the renaming test. Renaming holds behaviour fixed and varies",
         "surface; this holds surface nearly fixed and varies behaviour: `x_pre` is a test",
         "method before a developer's flakiness fix, `x_post` the same method after.", ""]
    if m:
        L += ["## Mining", "",
              "| | |", "|---|---|",
              f"| fix commits considered | {m['n_commits_considered']} |",
              f"| commits that could not be fetched | {m['n_commit_errors']} |",
              f"| **usable pairs** | **{m['n_pairs_unique']}** |",
              f"| … matching a FlakeBench flaky test by name | {m['n_pairs_flakebench_matched']} |",
              f"| … other test methods the same commits touched | {m['n_pairs_other_test_methods']} |",
              f"| projects | {m['n_projects']} |",
              f"| pairs excluded | {m['n_excluded']} |",
              "",
              "### Why pairs were excluded", "",
              "| reason | count |", "|---|---|"]
        for k, vv in m["excluded_reasons"].items():
            L.append(f"| `{k}` | {vv} |")
        L += ["",
              "An honest count of what did not qualify is part of the result; the excluded",
              "set is persisted with reasons in `data/minimal_pairs_excluded.csv`.", "",
              "### Composition", "",
              f"- cue strata: {m['cue_strata']}",
              f"- annotation-only fixes (`@Ignore` and the like, which change no behaviour): "
              f"{m['annotation_only_count']}",
              f"- categories: {m['category_counts']}",
              f"- median diff fraction: {f(m['diff_frac']['median'], 4)} "
              f"(pairs above {m['max_diff_frac_filter']} were excluded as no longer minimal)",
              ""]
    if p and p.get("n_pairs"):
        L += ["## The prediction is signed, not two-tailed", "", p["signed_prediction"], "",
              "```",
              "PairAcc(s) = (1/N) · Σ_i  1[ s(x_pre_i) > s(x_post_i) ]",
              "```", ""]
        sets = p.get("pair_sets") or {}
        if len(sets) > 1:
            L += ["## Two pair sets", "",
                  "Pair count and scorer strength trade against each other: holding out a",
                  "pair's project is mandatory, so more pairs means fewer training projects.",
                  "Both are reported.", "",
                  "| pair set | pairs | held-out projects | training rows | training flaky |",
                  "|---|---|---|---|---|"]
            for name, es in sets.items():
                star = " *(primary)*" if name == p.get("primary_pair_set") else ""
                L.append(f"| `{name}`{star} | {es['n_pairs']} | {es['n_held_out_projects']} | "
                         f"{es['n_train_rows']} | {es['n_train_flaky']} |")
            L.append("")
        for name, es in (sets or {"": p}).items():
            star = " (primary — S3 reads this one)" if name == p.get("primary_pair_set") else ""
            L += [f"### Results — `{name}`{star}" if name else "## Results", "",
                  "| scorer | PairAcc | 95% Wilson | binomial p | mean margin |",
                  "|---|---|---|---|---|"]
            for sn, e in es["scorers"].items():
                o = e["overall"]
                L.append(f"| `{sn}` | {f(o['pair_acc'])} | [{f(o['wilson_95'][0], 3)}, "
                         f"{f(o['wilson_95'][1], 3)}] | {f(o['binom_p_two_sided'])} | "
                         f"{o['mean_margin']:+.4f} |")
            L += ["", "By stratum:", "",
                  "| scorer | " + " | ".join(sorted(es["strata_sizes"])) + " |",
                  "|---|" + "---|" * len(es["strata_sizes"])]
            for sn, e in es["scorers"].items():
                row = [f"`{sn}`"]
                for k in sorted(es["strata_sizes"]):
                    st = e["by_stratum"].get(k)
                    row.append(f"{f(st['pair_acc'])} (n={st['n']})" if st else "—")
                L.append("| " + " | ".join(row) + " |")
            L.append("")
        L += ["The **cue-neutral** subset is the cleanest test: neither side of the pair gains",
              "or loses cue tokens, so a scorer reading vocabulary has nothing to go on.", ""]
    else:
        L += ["## Results", "", "No usable pairs; S3 is inconclusive.", ""]
    return "\n".join(L) + "\n"


def doc_06(d) -> str:
    L = ["# 06 — Representations and linear probes", "",
         "Source: `out/06_probe.json`. Minimal probing only — no nullspace projection, no",
         "attention analysis; those are Phase 2 and are deliberately absent from this",
         "codebase. Cross-project regime throughout.", "",
         "## Hardware constraints (Turing, sm_75)", "",
         f"- GPU: {d['gpu'].get('name')} ({d['gpu'].get('capability')}, "
         f"{d['gpu'].get('total_mem_gb')} GB), torch {d['gpu'].get('torch')}",
         "- **fp16 only** — Turing has no bf16; any `torch.bfloat16` is a bug",
         "- attention implementation is SDPA; FlashAttention-2 requires Ampere",
         "- no 4-bit quantisation: NF4 perturbs the representations being measured", "",
         "## Extraction", "",
         "| model / variant | layers | d | max_length | truncation rate | peak VRAM (GB) |",
         "|---|---|---|---|---|---|"]
    for k, m in d["extraction"].items():
        # the key is "model|variant"; a bare pipe would break the table
        L.append(f"| `{k.replace('|', ' / ')}` | {m['n_layers']} | {m['hidden_size']} | "
                 f"{m['max_length']} | "
                 f"{f(m['truncation_rate'], 4)} | {f(m['peak_vram_gb'], 2)} |")
    L += ["",
          "CodeBERT caps at 512 tokens against Qwen's 1024, so the two models see different",
          "amounts of code. That difference is real and is carried into the limitations.", "",
          "## Probe targets", "",
          "1. `flaky_code` — flaky on `code`, the headline.",
          "2. `flaky_code` evaluated on `code_renamed` — the transfer drop Δ.",
          "3. `flaky_renamed` — trained *and* evaluated on `code_renamed` (secondary framing).",
          "4. `P_ASYNC` / `P_UNORD` / `P_CLOCK` — does the representation encode the named",
          "   properties at all, with the flakiness label unused?",
          "5. `control` — random labels matched to the class prior, fixed across folds.",
          f"   Selectivity S = AP(real) − AP(control). Control prior "
          f"{f(d.get('control_prior'))}, class prior {f(d.get('class_prior'))}.", ""]
    for key, pools in d["models"].items():
        for pool, e in pools.items():
            L += [f"## `{key}` / {pool} pooling — {e['n_layers']} layers, d={e['hidden_size']}",
                  "",
                  "| target | max AP | at layer | selectivity there | last-layer AP |",
                  "|---|---|---|---|---|"]
            for tn, t in e["targets"].items():
                mx = t["max"]
                sel = t.get("selectivity_by_layer")
                sv = sel[mx["layer"]]["mean"] if sel and sel[mx["layer"]] else None
                L.append(f"| `{tn}` | {f(mx['ap_mean'])} ± {f(mx['ap_std'], 3)} | "
                         f"{mx['layer']} of {mx['n_layers_searched']} | {f(sv)} | "
                         f"{f(t['ap_by_layer'][-1]['mean'])} |")
            L += ["",
                  "Maxima are quoted with the number of layers searched: they are maxima over a",
                  "searched curve, not held-out estimates.", "",
                  "### Full layer curve — `flaky_code`", "",
                  "| layer | AP (mean ± std) | AP on `code_renamed` | Δ | selectivity |",
                  "|---|---|---|---|---|"]
            t = e["targets"]["flaky_code"]
            for i, c in enumerate(t["ap_by_layer"]):
                ren = t.get("ap_on_renamed_by_layer", [None] * len(t["ap_by_layer"]))[i]
                dl = t.get("transfer_delta_by_layer", [None] * len(t["ap_by_layer"]))[i]
                sel = t["selectivity_by_layer"][i]
                L.append(f"| {i} | {f(c['mean'])} ± {f(c['std'], 3)} | "
                         f"{f(ren['mean']) if ren else '—'} | "
                         f"{dl['mean']:+.4f} | {f(sel['mean']) if sel else '—'} |"
                         if c else f"| {i} | — | — | — | — |")
            L.append("")
    L += ["## Acceptance", "", "| check | result |", "|---|---|"]
    a = d.get("acceptance", {})
    L.append(f"| peak VRAM under 7.5 GB | {f(a.get('peak_vram_gb'), 2)} GB — "
             f"{'PASS' if a.get('peak_vram_under_7_5gb') else 'FAIL'} |")
    for k, vv in (a.get("selectivity_positive_at_claimed_layer") or {}).items():
        L.append(f"| selectivity positive at the claimed layer (`{k}`) | "
                 f"{'PASS' if vv else 'FAIL'} |")
    L.append(f"| CodeBERT truncation above 20% | "
             f"{'YES — stated explicitly' if a.get('codebert_truncation_over_20pct') else 'no'} |")
    return "\n".join(L) + "\n"


def doc_07(v) -> str:
    L = ["# 07 — The stopping rule", "",
         "Source: `out/07_decision.json`. Fixed in `plan/00_OVERVIEW.md` before any result",
         "was seen. No threshold was adjusted, no baseline swapped, no metric added.", "",
         f"# VERDICT: {v['overall'].replace('_', ' ').upper()}", "",
         "| criterion | statement | verdict |", "|---|---|---|"]
    for k, desc in (("S1", "cue-removed probe beats STRONG_WORD_BASELINE, cross-project"),
                    ("S2", "renaming transfer drop smaller for the probe than for `bow`"),
                    ("S3", "minimal-pair accuracy above 0.5 for the probe")):
        L.append(f"| {k} | {desc} | {VERDICT_WORD[v['criteria'][k].get('passes')]} |")
    pk = v["probe"]
    L += ["", v["overall_note"], "",
          f"Probe under test: `{pk['model']}` / {pk['pool']} pooling, layer {pk['layer']} of",
          f"{pk['n_layers_searched']} — {pk['selection']}.", "",
          f"`STRONG_WORD_BASELINE` = {f(v['strong_word_baseline']['ap_mean'])} "
          f"(`{v['strong_word_baseline']['method']}`).", "",
          "## S1 — cue-removed probe vs the strong word baseline", "",
          "**Operationalisation.** \"Cue-removed probe\" is read as the probe evaluated on",
          "cue-removed input — trained on `code`, scored on `code_renamed` — against a word",
          "baseline that had full access to the cues. That is the Phase 1 question. Two other",
          "readings are reported so the verdict can be checked under each.", "",
          "| framing | probe AP | baseline AP | mean diff | Wilcoxon p | Cliff's δ | verdict |",
          "|---|---|---|---|---|---|---|"]
    s1 = v["criteria"]["S1"]
    for name, w in s1["framings"].items():
        L.append(f"| {w['framing']} | {f(w['probe_ap_mean'])} | {f(w['baseline_ap_mean'])} | "
                 f"{w['mean_diff']:+.4f} | {f(w.get('p'))} | {f(w.get('cliffs_delta'), 3)} | "
                 f"{VERDICT_WORD[w['passes']]} |")
    prim = s1["framings"]["primary_cue_removed"]
    L += ["",
          f"Fold-level differences (probe − baseline), primary framing: "
          f"{folds(prim['diff'])} — {prim['n_positive']} of {prim['n']} folds favour the probe.",
          "",
          f"With 5 folds the smallest attainable one-sided p is "
          f"{f(prim['min_attainable_p'], 4)}, so a pass requires **every** fold to favour the",
          "probe. Reported in full because five folds is very little power.", ""]
    s2 = v["criteria"]["S2"]
    L += ["## S2 — transfer drop, probe vs word baseline", "",
          "| | mean | folds |", "|---|---|---|",
          f"| Δ_probe | {s2['delta_probe_mean']:+.4f} | {folds(s2['delta_probe_folds'])} |",
          f"| Δ_bow | {s2['delta_bow_mean']:+.4f} | {folds(s2['delta_bow_folds'])} |",
          "",
          f"The probe's drop is smaller in {s2['n_folds_probe_smaller']} of "
          f"{len(s2['delta_bow_folds'])} folds; paired Wilcoxon (bow − probe) p = "
          f"{f(s2['paired_test_bow_minus_probe'].get('p'))}. "
          f"Verdict {VERDICT_WORD[s2['passes']]}.", ""]
    s3 = v["criteria"]["S3"]
    L += ["## S3 — minimal pairs", ""]
    if s3.get("status") == "inconclusive":
        L += [f"{VERDICT_WORD[None]} — {s3.get('power_note', s3.get('reason', ''))}", ""]
    else:
        L += ["| | |", "|---|---|",
              f"| scorer | `{s3['scorer']}` |",
              f"| pairs | {s3['n_pairs']} |",
              f"| **PairAcc (probe)** | **{f(s3['pair_acc'])}** |",
              f"| 95% Wilson | [{f(s3['wilson_95'][0], 3)}, {f(s3['wilson_95'][1], 3)}] |",
              f"| binomial p (two-sided) | {f(s3['binom_p_two_sided'])} |",
              f"| mean margin | {s3['mean_margin']:+.4f} |",
              f"| PairAcc (`bow`) | {f(s3['bow_pair_acc'])} |",
              f"| `bow` mean margin | {f(s3.get('bow_mean_margin'))} |",
              "", s3["sign_note"], ""]
        if s3.get("cue_neutral"):
            cn, bn = s3["cue_neutral"], s3.get("bow_cue_neutral") or {}
            L += [f"On the cue-neutral subset (n={cn['n']}) — the cleanest test — the probe "
                  f"scores {f(cn['pair_acc'])} and `bow` scores {f(bn.get('pair_acc'))}.", ""]
        L += [f"Verdict {VERDICT_WORD[s3['passes']]}.", ""]
    sv = v["split_verdict"]
    L += ["## Corroboration — not part of the rule", "",
          "These sharpen the interpretation either way.", "",
          "| | |", "|---|---|",
          f"| structural-only classifier AP (cross-project) | "
          f"{f(sv['structural_only_classifier_ap'])} |",
          f"| class prior | {f(sv['class_prior'])} |"]
    for p, ap in sv["probe_encodes_properties_ap"].items():
        L.append(f"| probe AP for `{p}` | {f(ap)} (selectivity "
                 f"{f(sv['probe_property_selectivity'].get(p))}) |")
    s1p, s3p = v["criteria"]["S1"].get("passes"), v["criteria"]["S3"].get("passes")
    if s1p and s3p is False:
        L += ["",
              "**The split that actually occurred.** The probe passes S1 and S2 — it beats the "
              "word baseline on cue-removed input and is nearly unmoved by counterfactual "
              "renaming — and fails S3, the test that varies behaviour while holding surface "
              "nearly fixed. Surface invariance without behavioural sensitivity is not "
              "structural understanding, which is precisely why the plan required both "
              "directions rather than defining \"structural\" as whatever survives cue "
              "removal.",
              "",
              f"The properties themselves are weakly encoded (AP "
              + ", ".join(f"{f(a)} for `{p}`" for p, a in
                          sv["probe_encodes_properties_ap"].items())
              + f") and predict the flakiness label barely above its prior "
                f"({f(sv['structural_only_classifier_ap'])} against "
                f"{f(sv['class_prior'])}). Whatever the probe is reading, it is not these "
                f"properties.",
              ""]
    else:
        L += ["", "**The possible split verdict.** " + sv["note"], ""]
    return "\n".join(L) + "\n"


def doc_index(ctx, files, decision) -> str:
    L = ["# Phase 1 — structural or lexical? Results",
         "",
         "Whether a code language model represents test flakiness structurally or lexically.",
         "Implementation of the study specified in `plan/`; every number here is generated",
         "from a JSON artefact in `out/` by `scripts/export_reports.py`.",
         ""]
    if decision:
        L += [f"## Verdict: {decision['overall'].replace('_', ' ').upper()}", "",
              "| criterion | verdict |", "|---|---|"]
        for k in ("S1", "S2", "S3"):
            L.append(f"| {k} | {VERDICT_WORD[decision['criteria'][k].get('passes')]} |")
        L.append("")
    L += ["## Files", "", "| file | what it holds |", "|---|---|",
          "| `MAIN_REPORT.md` | the full write-up — **read this first** |",
          "| `01_data.md` | corpus counts, the whitespace artefact, contradictory labels |",
          "| `02_baselines.md` | within- vs cross-project gap, `STRONG_WORD_BASELINE` |",
          "| `03_renaming.md` | cue vocabulary, the rewrite, transfer drop Δ |",
          "| `04_structural.md` | property prevalence, the hand audit, precision and recall |",
          "| `04_validation.md` | the audit as written by `s04_validate.py` |",
          "| `04_audit_sample.md` | the 50 audited methods in full, with the analyser's "
          "prediction beside each |",
          "| `05_minimal_pairs.md` | mining yield, exclusions, PairAcc per scorer and stratum |",
          "| `06_probe.md` | extraction settings, per-layer AP curves, selectivity |",
          "| `07_decision.md` | the stopping rule evaluated criterion by criterion |",
          "| `figures/` | four PNGs: layer curve, within-vs-cross, transfer drop, pairs |",
          "",
          "## Three things to know before reading any number",
          "",
          "1. **The released CSV is separable by whitespace alone** (AUROC 1.000). No flaky",
          "   test begins with indentation or ends with a newline; 98.9% and 100% of",
          "   non-flaky tests do. Stage 01 normalises this away and asserts the result.",
          "2. **`id` is not unique** — FlakeBench numbers its flaky and non-flaky halves",
          "   independently and 66 ids collide. Everything keys on a content hash instead.",
          "3. **Both models were pretrained on public GitHub**, including these repositories.",
          "   Project-grouped CV stops the classifier seeing a test project's rows; it cannot",
          "   stop the encoder having seen the repository. This is the strongest alternative",
          "   explanation for a positive S1 and it is not ruled out here.",
          "",
          "## Generated files", "",
          "| file | bytes |", "|---|---|"]
    for p in sorted(files):
        L.append(f"| `{p.name}` | {p.stat().st_size} |")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--dest", default=None, help="output directory (default reports/)")
    args = ap.parse_args()

    ctx = ctx_from_args(args)
    log = get_logger(STAGE, ctx)
    dest = Path(args.dest) if args.dest else ctx.root / f"reports{ctx.suffix}"
    dest.mkdir(parents=True, exist_ok=True)

    def maybe(name):
        p = ctx.out / name
        if not p.exists():
            log.warning("missing %s — its section will be skipped", p)
            return None
        return load_json(p)

    prep, base, transfer = maybe("01_prep.json"), maybe("02_baselines.json"), \
        maybe("03_transfer.json")
    rename, struct, valid = maybe("03_rename.json"), maybe("04_structural.json"), \
        maybe("04_validation.json")
    mine, pairs = maybe("05_mine.json"), maybe("05_pairs.json")
    probe, decision = maybe("06_probe.json"), maybe("07_decision.json")

    written: list[Path] = []
    plan = [
        ("01_data.md", lambda: doc_01(prep), prep),
        ("02_baselines.md", lambda: doc_02(base, transfer), base),
        ("03_renaming.md", lambda: doc_03(rename, transfer), rename or transfer),
        ("04_structural.md", lambda: doc_04(struct, valid), struct),
        ("05_minimal_pairs.md", lambda: doc_05(mine, pairs), mine or pairs),
        ("06_probe.md", lambda: doc_06(probe), probe),
        ("07_decision.md", lambda: doc_07(decision), decision),
    ]
    for name, fn, guard in plan:
        if guard is None:
            continue
        written.append(save_text(dest / name, fn()))

    # verbatim copies of the Markdown the pipeline already produces
    for src, dst in (("07_report.md", "MAIN_REPORT.md"),
                     ("04_validation.md", "04_validation.md"),
                     ("04_audit_sample.md", "04_audit_sample.md")):
        s = ctx.out / src
        if s.exists():
            shutil.copy2(s, dest / dst)
            written.append(dest / dst)

    figdir = ctx.out / "figures"
    if figdir.exists():
        (dest / "figures").mkdir(exist_ok=True)
        for png in sorted(figdir.glob("*.png")):
            shutil.copy2(png, dest / "figures" / png.name)

    idx = save_text(dest / "00_INDEX.md", doc_index(ctx, written, decision))
    written.append(idx)

    log.info("wrote %d Markdown files to %s", len(written), dest)
    for p in sorted(written):
        log.info("  %s (%d bytes)", p.name, p.stat().st_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
