# 02 — Baselines and the within- vs cross-project gap

Replicates the generalisation gap reported by Haben et al. (MSR 2021) on this
dataset, and fixes the baseline that everything later must beat.

## Two evaluation regimes

Both use 5 folds, stratified, seed 0.

- **within**: `StratifiedKFold`. Tests from one project appear in both train and
  test. Measures interpolation.
- **cross**: `StratifiedGroupKFold` grouped by `project`. No project spans the
  split. Measures generalisation.

Report `gap = AP_within − AP_cross` for every method. A large gap is itself a
result and belongs in the write-up regardless of what happens later.

## Baselines

Fit each under both regimes, on the full set and on the `matched` subset.

| name | features |
|------|----------|
| `length` | `n_chars`, `n_lines`, `n_tokens` |
| `bow` | TF-IDF over Java identifiers, `min_df=3`, case preserved, sublinear tf |
| `bow_ablated` | `bow` with the cue vocabulary from `03` removed |
| `char_ngram` | TF-IDF char 3–5 grams |
| `bow_plus_length` | `bow` ⊕ standardised length features |

Classifier: `LogisticRegression(class_weight='balanced', max_iter=2000)`, C
tuned on an inner split, not on test folds. Vectorisers fit **inside** each fold.

Case preserved matters — camelCase carries signal in Java identifiers, and
lowercasing destroys it.

## The number that matters

`STRONG_WORD_BASELINE` = best cross-project AP over `{bow, char_ngram,
bow_plus_length}`. Write it to `out/02_baselines.json`. Stopping rule S1 is
evaluated against this value, so it is fixed here and not renegotiated later.

## Output

`out/02_baselines.json`: per method, per regime, per subset — mean and std AP
and AUROC across folds, plus the fold-level vectors (needed for paired tests in
`07`), plus `STRONG_WORD_BASELINE`.

## Acceptance

- Every method scores above the class prior under both regimes.
- `gap > 0` for `bow`. If cross ≥ within, the grouping is broken — check that
  `project` is actually being passed as `groups`.
- Fold-level AP vectors are persisted, not just summary statistics.
