# Phase 1 — structural or lexical? Results

Whether a code language model represents test flakiness structurally or lexically.
Implementation of the study specified in `plan/`; every number here is generated
from a JSON artefact in `out/` by `scripts/export_reports.py`.

## Verdict: STOP AND WRITE UP

| criterion | verdict |
|---|---|
| S1 | **PASS** |
| S2 | **PASS** |
| S3 | **FAIL** |

## Files

| file | what it holds |
|---|---|
| `MAIN_REPORT.md` | the full write-up — **read this first** |
| `01_data.md` | corpus counts, the whitespace artefact, contradictory labels |
| `02_baselines.md` | within- vs cross-project gap, `STRONG_WORD_BASELINE` |
| `03_renaming.md` | cue vocabulary, the rewrite, transfer drop Δ |
| `04_structural.md` | property prevalence, the hand audit, precision and recall |
| `04_validation.md` | the audit as written by `s04_validate.py` |
| `04_audit_sample.md` | the 50 audited methods in full, with the analyser's prediction beside each |
| `05_minimal_pairs.md` | mining yield, exclusions, PairAcc per scorer and stratum |
| `06_probe.md` | extraction settings, per-layer AP curves, selectivity |
| `07_decision.md` | the stopping rule evaluated criterion by criterion |
| `figures/` | four PNGs: layer curve, within-vs-cross, transfer drop, pairs |

## Three things to know before reading any number

1. **The released CSV is separable by whitespace alone** (AUROC 1.000). No flaky
   test begins with indentation or ends with a newline; 98.9% and 100% of
   non-flaky tests do. Stage 01 normalises this away and asserts the result.
2. **`id` is not unique** — FlakeBench numbers its flaky and non-flaky halves
   independently and 66 ids collide. Everything keys on a content hash instead.
3. **Both models were pretrained on public GitHub**, including these repositories.
   Project-grouped CV stops the classifier seeing a test project's rows; it cannot
   stop the encoder having seen the repository. This is the strongest alternative
   explanation for a positive S1 and it is not ruled out here.

## Generated files

| file | bytes |
|---|---|
| `01_data.md` | 3269 |
| `02_baselines.md` | 2507 |
| `03_renaming.md` | 3789 |
| `04_audit_sample.md` | 77130 |
| `04_structural.md` | 4669 |
| `04_validation.md` | 3495 |
| `05_minimal_pairs.md` | 4312 |
| `06_probe.md` | 7980 |
| `07_decision.md` | 3854 |
| `MAIN_REPORT.md` | 22494 |
