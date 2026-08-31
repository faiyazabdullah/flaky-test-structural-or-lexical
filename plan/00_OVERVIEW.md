# 00 — Overview

Phase 1 of a study on whether code language models represent test flakiness
structurally or lexically. Scope was set by the supervisor: establish the
within- vs cross-project gap, run cue removal and counterfactual renaming, test
minimal pairs from fix commits, and stop if the signal is lexical. Probing with
nullspace projection and attention analysis are explicitly **out of scope** and
must not be implemented here.

## What Phase 1 answers

> After removing lexical cues, does any predictive signal for flakiness survive
> across unseen projects — and is that signal attributable to named structural
> properties rather than to project or dataset artefacts?

## Definition being used

"Structural information" is **not** defined as residue after cue removal. That
definition is circular: signal can survive because of cues that were missed,
framework idiom, or label-collection artefacts. Two replacements are used
instead, and both must be implemented.

**Two-sided invariance.** The existing renaming test checks that predictions are
stable when surface changes and behaviour does not. Phase 1 adds the converse:
predictions must *change* when behaviour changes and surface barely does. Fix
commits supply those pairs.

**Named properties with independent ground truth.** Three properties are labelled
by static analysis, never by the flakiness label:

| id | property |
|----|----------|
| `P_ASYNC` | async dispatch reaches an assertion with no intervening wait |
| `P_UNORD` | value from unordered iteration reaches an assertion |
| `P_CLOCK` | clock read reaches an assertion |

A fourth property — shared field written by one test and read by another — is
inter-procedural and **not computable** from FlakeBench's isolated method
bodies. It is deferred to Phase 2, which requires repository checkout.

## Stopping rule — fixed in advance, do not revise after seeing results

Proceed to Phase 2 only if **all three** hold:

- **S1** Cue-removed probe beats the strongest word-based baseline under
  project-grouped CV. Paired one-sided test over folds, α = 0.05.
- **S2** Renaming transfer drop is smaller for the probe than for the word
  baseline.
- **S3** Minimal-pair accuracy is above 0.5 for the probe, binomial test,
  α = 0.05.

If S1 fails, stop and write the negative result. That outcome is a deliverable,
not a failure, and `07_DECISION.md` covers how to report it.

## Hardware

NVIDIA RTX 2070, 8GB, Turing (sm_75).

- **fp16 only.** Turing has no bf16. Any `torch.bfloat16` in generated code is a
  bug.
- **No FlashAttention-2** (Ampere+). Use SDPA or eager.
- Models: `microsoft/codebert-base` (125M) and
  `Qwen/Qwen2.5-Coder-1.5B` (~3.1GB fp16). Both fit with room at 1024 tokens.
- `Qwen/Qwen2.5-Coder-3B` (~6.2GB fp16) is a stretch target at batch size 1 only.
- Do **not** use 4-bit quantisation in Phase 1. NF4 perturbs the representations
  being measured and confounds the result.

## Conventions

- Python 3.10+, `torch`, `transformers`, `scikit-learn`, `pandas`,
  `tree-sitter`, `tree-sitter-java`.
- Layout: `data/` (derived), `out/` (arrays, results), `scripts/`, `tests/`.
- Every script takes `--seed`, defaults to 0, and is deterministic.
- Every script writes a JSON result file to `out/` alongside stdout.
- **Primary metric is average precision (AP).** Never report accuracy on the
  imbalanced full set.
- **Never use a random split for a cross-project claim.** Grouped by project
  throughout, except where `02` deliberately contrasts the two.
- Write unit tests for the renaming and structural-analysis passes. Those two
  are where silent bugs will invalidate everything downstream.

## Build order

`01` → `02` → `03` → `04` → `05` → `06` → `07`. Files `04` and `05` are
independent of each other and can be built in either order.
