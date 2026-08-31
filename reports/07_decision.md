# 07 — The stopping rule

Source: `out/07_decision.json`. Fixed in `plan/00_OVERVIEW.md` before any result
was seen. No threshold was adjusted, no baseline swapped, no metric added.

# VERDICT: STOP AND WRITE UP

| criterion | statement | verdict |
|---|---|---|
| S1 | cue-removed probe beats STRONG_WORD_BASELINE, cross-project | **PASS** |
| S2 | renaming transfer drop smaller for the probe than for `bow` | **PASS** |
| S3 | minimal-pair accuracy above 0.5 for the probe | **FAIL** |

All three must hold to proceed to Phase 2. Otherwise stop and write up. The negative write-up is a deliverable, not a fallback.

Probe under test: `qwen1_5b` / last pooling, layer 6 of
29 — argmax of flaky_code cross-project AP; a maximum over a searched curve, not a held-out estimate.

`STRONG_WORD_BASELINE` = 0.2032 (`char_ngram`).

## S1 — cue-removed probe vs the strong word baseline

**Operationalisation.** "Cue-removed probe" is read as the probe evaluated on
cue-removed input — trained on `code`, scored on `code_renamed` — against a word
baseline that had full access to the cues. That is the Phase 1 question. Two other
readings are reported so the verdict can be checked under each.

| framing | probe AP | baseline AP | mean diff | Wilcoxon p | Cliff's δ | verdict |
|---|---|---|---|---|---|---|
| probe trained on `code`, evaluated on `code_renamed` | 0.7031 | 0.2032 | +0.5000 | 0.0312 | 1.000 | **PASS** |
| probe trained and evaluated on `code` | 0.7085 | 0.2032 | +0.5053 | 0.0312 | 1.000 | **PASS** |
| probe trained and evaluated on `code_renamed` | 0.7102 | 0.2032 | +0.5070 | 0.0312 | 1.000 | **PASS** |

Fold-level differences (probe − baseline), primary framing: 0.394, 0.447, 0.374, 0.661, 0.624 — 5 of 5 folds favour the probe.

With 5 folds the smallest attainable one-sided p is 0.0312, so a pass requires **every** fold to favour the
probe. Reported in full because five folds is very little power.

## S2 — transfer drop, probe vs word baseline

| | mean | folds |
|---|---|---|
| Δ_probe | +0.0053 | 0.005, 0.002, 0.008, 0.005, 0.007 |
| Δ_bow | +0.0617 | 0.095, 0.054, 0.081, 0.012, 0.066 |

The probe's drop is smaller in 5 of 5 folds; paired Wilcoxon (bow − probe) p = 0.0312. Verdict **PASS**.

## S3 — minimal pairs

| | |
|---|---|
| scorer | `probe_qwen1_5b_last` |
| pairs | 50 |
| **PairAcc (probe)** | **0.4200** |
| 95% Wilson | [0.294, 0.558] |
| binomial p (two-sided) | 0.3222 |
| mean margin | -0.0857 |
| PairAcc (`bow`) | 0.4800 |
| `bow` mean margin | 0.0363 |

The contrast with bow is the substantive result: a cue-riding model is expected below 0.5 because fixes add cue tokens.

On the cue-neutral subset (n=33) — the cleanest test — the probe scores 0.3333 and `bow` scores 0.4848.

Verdict **FAIL**.

## Corroboration — not part of the rule

These sharpen the interpretation either way.

| | |
|---|---|
| structural-only classifier AP (cross-project) | 0.0460 |
| class prior | 0.0325 |
| probe AP for `P_ASYNC` | 0.2070 (selectivity 0.1707) |
| probe AP for `P_UNORD` | 0.1565 (selectivity 0.1202) |
| probe AP for `P_CLOCK` | 0.2104 (selectivity 0.1656) |

**The split that actually occurred.** The probe passes S1 and S2 — it beats the word baseline on cue-removed input and is nearly unmoved by counterfactual renaming — and fails S3, the test that varies behaviour while holding surface nearly fixed. Surface invariance without behavioural sensitivity is not structural understanding, which is precisely why the plan required both directions rather than defining "structural" as whatever survives cue removal.

The properties themselves are weakly encoded (AP 0.2070 for `P_ASYNC`, 0.1565 for `P_UNORD`, 0.2104 for `P_CLOCK`) and predict the flakiness label barely above its prior (0.0460 against 0.0325). Whatever the probe is reading, it is not these properties.

