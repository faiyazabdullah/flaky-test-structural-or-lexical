# 03 — Cue removal and counterfactual renaming

Sources: `out/03_rename.json`, `out/03_transfer.json`, `data/cue_vocab.json`,
`data/rename_map.json`.

Two interventions on the lexical channel:

- **A — removal.** Cue tokens dropped from the vectoriser vocabulary. Feature space
  only; the code is untouched. Feeds `bow_ablated` in 02.
- **B — counterfactual renaming.** The source is rewritten, each cue replaced by a
  neutral identifier of matched shape (`Thread.sleep` → `Util.pause`,
  `CountDownLatch` → `Gate`). Control flow, call structure, arity and assertions
  are preserved.

## Cue vocabulary

Three separately-tagged sources, so each can be ablated alone.

| tag | size | provenance |
|---|---|---|
| `attributed` | 4 | tokens the plan cites from Rahman et al. (OOPSLA'25) |
| `attributed_curated` | 45 | per-category extension curated in this repository, **not** from the paper |
| `api` | 81 | nondeterminism-related API and type names |
| `mined_audit_only` | 200 | top identifiers by mutual information with the label, on the full set — **audit artefact only, never used to fit anything** |

Only the label-independent tags drive the renaming. Mined cues correlate with
the label by construction, so building the rewritten input from them would leak
the label into the representation; they are used only for intervention A, and
there they are re-mined inside each training fold.

## Shape preservation

- rows touched: 2394 (28.4%), mean 0.97 replacements per row
- **Java token count identical on 1.0000 of rows** — every substitution swaps one identifier token for one identifier token, so this is exact by construction (acceptance: ≥ 0.95)
- token *kinds* identical on 1.0000 of rows
- CodeBERT subword counts still shift (mean +0.59, unchanged on 0.785 of rows) because a neutral name segments differently — reported rather than hidden

### Most-replaced cues

| corpus occurrences | cue | neutral name |
|---|---|---|
| 777 | `System` | `SymBp` |
| 565 | `iterator` | `symCr` |
| 501 | `HashMap` | `MapA` |
| 418 | `TimeUnit` | `Scale` |
| 378 | `Thread` | `SymBq` |
| 373 | `now` | `symC1` |
| 316 | `Duration` | `SymY` |
| 285 | `Instant` | `Sym8` |
| 284 | `CountDownLatch` | `Gate` |
| 264 | `InterruptedException` | `HaltedFailure` |
| 253 | `await` | `stall` |
| 238 | `sleep` | `symDd` |
| 237 | `values` | `secondView` |
| 231 | `execute` | `invokeOn` |
| 207 | `Calendar` | `Almanac` |

## Transfer drop

Δ = AP(`code`) − AP(`code_renamed`), cross-project, model **trained on `code`**.
The asymmetry between the train and eval distributions is the whole
measurement.

| method | AP on `code` | AP on `code_renamed` | Δ | Δ folds |
|---|---|---|---|---|
| `length` | 0.0892 | 0.0892 | +0.0000 | 0.000, 0.000, 0.000, 0.000, 0.000 |
| `bow` | 0.1488 | 0.0872 | +0.0617 | 0.095, 0.054, 0.081, 0.012, 0.066 |
| `bow_ablated` | 0.0828 | 0.0880 | -0.0053 | -0.005, -0.015, -0.005, -0.001, -0.001 |
| `char_ngram` | 0.2032 | 0.1808 | +0.0224 | 0.068, 0.021, -0.001, 0.008, 0.017 |
| `bow_plus_length` | 0.1746 | 0.1129 | +0.0617 | 0.068, 0.065, 0.071, 0.037, 0.068 |

### Sanity check: retraining on the renamed variant

A `bow` model **retrained** on `code_renamed` scores 0.1489, against 0.1488 on `code` — a difference of 0.0001, within one fold-std (0.0382): **yes**.

Retraining measures nothing about lexical reliance — a bag-of-tokens model
simply relearns `pause` and `Gate`. It is reported only as evidence that
the renaming is shape-preserving and not leaking information.

## Acceptance

| check | result |
|---|---|
| rename_unit_tests | see tests/test_rename.py -- run via scripts/run_all.py --tests |
| java_token_count_equal_ge_95pct | PASS |
| rename_map_injective | PASS |
| no_collision_with_corpus_identifiers | PASS |
