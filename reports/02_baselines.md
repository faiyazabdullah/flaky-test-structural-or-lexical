# 02 — Baselines and the within- vs cross-project gap

Source: `out/02_baselines.json`. 5 folds, stratified, seed 0. `within` uses
`StratifiedKFold`; `cross` uses `StratifiedGroupKFold` grouped by project, so no
project spans the split. Vectorisers are fitted inside each fold and `C` is tuned
on an inner split of the training rows.

Class prior: 0.0325 (full set), 0.5000 (matched subset).

## Full set

| method | within AP | cross AP | gap | within AUROC | cross AUROC |
|---|---|---|---|---|---|
| `length` | 0.0910 ± 0.007 | 0.0892 ± 0.038 | +0.0019 | 0.7298 | 0.7258 |
| `bow` | 0.5027 ± 0.051 | 0.1488 ± 0.038 | +0.3538 | 0.9069 | 0.7508 |
| `bow_ablated` | 0.3659 ± 0.037 | 0.0828 ± 0.016 | +0.2832 | 0.8675 | 0.6943 |
| `char_ngram` | 0.5539 ± 0.080 | 0.2032 ± 0.104 | +0.3508 | 0.9398 | 0.8174 |
| `bow_plus_length` | 0.5350 ± 0.060 | 0.1746 ± 0.047 | +0.3604 | 0.9151 | 0.7780 |

## Length-matched subset

| method | within AP | cross AP | gap |
|---|---|---|---|
| `length` | 0.6811 | 0.6769 | +0.0042 |
| `bow` | 0.7518 | 0.6465 | +0.1053 |
| `bow_ablated` | 0.6906 | 0.5964 | +0.0942 |
| `char_ngram` | 0.8124 | 0.6805 | +0.1319 |
| `bow_plus_length` | 0.8144 | 0.7276 | +0.0868 |

## Fold-level cross-project AP (full set)

Reported in full because five folds is very little power and the paired tests in
07 read these vectors directly.

| method | fold 1 | fold 2 | fold 3 | fold 4 | fold 5 |
|---|---|---|---|---|---|
| `length` | 0.0988 | 0.0782 | 0.0525 | 0.1497 | 0.0666 |
| `bow` | 0.2041 | 0.1093 | 0.1644 | 0.1177 | 0.1487 |
| `bow_ablated` | 0.0752 | 0.0856 | 0.0639 | 0.1062 | 0.0830 |
| `char_ngram` | 0.3897 | 0.1583 | 0.1564 | 0.1487 | 0.1630 |
| `bow_plus_length` | 0.1629 | 0.1337 | 0.1525 | 0.2558 | 0.1679 |

## STRONG_WORD_BASELINE

**0.2032** — `char_ngram`, cross-project, full set. Fixed here
and not renegotiated; stopping rule S1 is evaluated against it.

| candidate | cross AP |
|---|---|
| `bow` | 0.1488 |
| `char_ngram` | 0.2032 |
| `bow_plus_length` | 0.1746 |

## Cue ablation

`bow_ablated` is `bow` with the cue vocabulary removed from the feature space
(the code is untouched). Mined cues are re-mined inside each training fold.

- `bow` cross-project AP 0.1488 → `bow_ablated` 0.0828 (**-0.0660**)
- features dropped per fold: [271, 265, 276, 265, 271] of [9274, 9021, 9503, 9115, 9066]

## Acceptance

| check | result |
|---|---|
| all_methods_above_class_prior | PASS |
| bow_gap_positive | PASS |
| fold_level_vectors_persisted | PASS |
