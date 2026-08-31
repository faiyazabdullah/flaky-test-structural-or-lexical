# 01 — Data preparation

Source: `out/01_prep.json`. Produces `data/prepped.csv`, whose row order every
later stage assumes.

## Counts

| | |
|---|---|
| raw rows | 8574 |
| exact duplicates removed | 139 |
| duplicates removed after normalisation | 1 |
| rows dropped for contradictory labels | 12 |
| **final rows** | **8422** |
| flaky | 274 (0.0325) |
| projects | 98 |
| source `id` values that collide | 66 |

## Class composition

| label | rows |
|---|---|
| `non-flaky` | 8148 |
| `test order dependency` | 87 |
| `async wait` | 76 |
| `unordered collections` | 41 |
| `concurrency` | 37 |
| `time` | 33 |

## The formatting artefact

In the released CSV the two classes are separable by whitespace alone.

| feature | non-flaky | flaky |
|---|---|---|
| `ws_leading_indent` | 0.9891 | 0.0000 |
| `ws_ends_with_newline` | 1.0000 | 0.0000 |
| `ws_contains_tab` | 0.1101 | 0.0000 |
| `ws_contains_crlf` | 0.0178 | 0.0000 |

A logistic regression on those four booleans gives **AUROC 1.000** before normalisation and **0.500** after.

The classes were evidently extracted by different pipelines. A subword tokenizer
encodes leading whitespace, so without normalisation every downstream number
measures provenance rather than flakiness.

## Length confound

- median chars: flaky 811 vs non-flaky 446
- `n_tokens` alone reaches AUROC **0.6882** on the full set and **0.4997** on the length-matched subset
- matched subset: 548 rows (274/274), median `n_tokens` 285 vs 284 (relative difference 0.0018)

## Contradictory labels

6 test methods appear twice with **identical code and opposite labels** (12 rows). Neither label can be trusted, so both copies were dropped rather than one being silently preferred.

| id | project | test | label |
|---|---|---|---|
| 62 | `wildfly_wildfly` | `b19048b72669fc0e96665b1b125dc1fda21f5993.testJavaContext` | `test order dependency` |
| 74 | `wildfly_wildfly` | `b19048b72669fc0e96665b1b125dc1fda21f5993.testBind.2` | `test order dependency` |
| 79 | `wildfly_wildfly` | `b19048b72669fc0e96665b1b125dc1fda21f5993.testRejectionEAP6` | `test order dependency` |
| 166 | `wildfly_wildfly` | `b19048b72669fc0e96665b1b125dc1fda21f5993.testRejectionEAP7` | `test order dependency` |
| 169 | `wildfly_wildfly` | `b19048b72669fc0e96665b1b125dc1fda21f5993.testBindNested` | `test order dependency` |
| 268 | `wildfly_wildfly` | `b19048b72669fc0e96665b1b125dc1fda21f5993.testRebind.2` | `test order dependency` |
| 26836 | `wildfly_wildfly` | `WritableServiceBasedNamingStoreTestCase.testBind` | `non-flaky` |
| 26837 | `wildfly_wildfly` | `WritableServiceBasedNamingStoreTestCase.testBindNested` | `non-flaky` |
| 26842 | `wildfly_wildfly` | `WritableServiceBasedNamingStoreTestCase.testRebind` | `non-flaky` |
| 26851 | `wildfly_wildfly` | `NamingSubsystemTestCase.testRejectionsEAP7` | `non-flaky` |
| 26852 | `wildfly_wildfly` | `NamingSubsystemTestCase.testRejectionsEAP6` | `non-flaky` |
| 26906 | `wildfly_wildfly` | `InitialContextFactoryTestCase.testJavaContext` | `non-flaky` |

## Acceptance

| check | result |
|---|---|
| no_duplicate_code | PASS |
| uid_unique | PASS |
| leading_indent_rate_zero | PASS |
| whitespace_auroc_within_tolerance | PASS |
| matched_balanced | PASS |
| matched_median_within_2pct | PASS |
