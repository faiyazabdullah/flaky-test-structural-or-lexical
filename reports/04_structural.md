# 04 — Structural property labels

Sources: `out/04_structural.json`, `out/04_validation.json`. Labels come from a
tree-sitter taint pass and never from the flakiness label — that is what makes
"structural information" a testable claim rather than a residue left over after
cue removal.

| property | definition |
|---|---|
| `P_ASYNC` | an async dispatch reaches an assertion with no intervening synchronisation on the path |
| `P_UNORD` | a value originating from unordered iteration reaches an assertion argument |
| `P_CLOCK` | a clock read reaches an assertion argument |

A fourth property — a shared field written by one test and read by another — is
inter-procedural, needs the class body and sibling tests, and **cannot** be
computed from FlakeBench's isolated method bodies. Deferred to Phase 2.

## Parsing

- parse failure rate **0.0241** (acceptance < 0.05)
- without truncation repair it would be 0.0649
- rows repaired: 344 (4.1%) — {'truncate_to_last_statement': 172, 'rewind_and_close': 147, 'close_brackets': 19, 'semicolon': 6}

FlakeBench truncates a fraction of method bodies mid-statement. Those rows are
almost entirely non-flaky, so leaving them unparsed would bias the labels in
exactly the direction the plan warns about. The repair closes open brackets or
rewinds to the last complete statement; it never invents content, and every
repaired row is flagged in `data/structural_full.csv`.

Remaining failures by label: {'non-flaky': 200, 'time': 3}

## Prevalence

| property | overall | async wait | concurrency | non-flaky | test order dependency | time | unordered collections |
|---|---|---|---|---|---|---|---|
| `P_ASYNC` | 0.0287 | 0.1316 | 0.1351 | 0.0269 | 0.0460 | 0.0909 | 0.0244 |
| `P_UNORD` | 0.0099 | 0.0132 | 0.0000 | 0.0099 | 0.0000 | 0.0000 | 0.0244 |
| `P_CLOCK` | 0.0156 | 0.0263 | 0.0000 | 0.0146 | 0.0000 | 0.3030 | 0.0000 |

**Floor check** (not a finding): `P_ASYNC` prevalence in the async-wait category 0.1316 vs non-flaky 0.0269 — passes.

## A consequence of the definition

The plan puts `sleep` in the synchronisation kill set, so the canonical
async-wait flaky shape — *dispatch; sleep; assert* — is excluded from
`P_ASYNC` by construction.

- rows that would flip if `sleep` were not a barrier: 20
- async-wait rate would rise from 0.1316 to 0.1842

Diagnostic only. No stopping-rule number depends on it.

## Do the structural properties predict flakiness?

A logistic regression on the three booleans alone, cross-project:

- **AP 0.0460 ± 0.0108** (folds: 0.030, 0.054, 0.039, 0.054, 0.053)
- class prior 0.0325

Barely above the prior. On this dataset the named structural properties carry
almost no predictive signal for the flakiness label — which is reportable either
way, and is the first half of the possible split verdict in 07.

## Validation against the hand audit

50 methods, stratified by property and by label, read in full
by Claude (implementing agent). Precision floor **0.8**: below it a
property is either fixed or dropped from the study.

### Definitional — does the property hold as the plan writes it?

| property | TP | FP | FN | TN | precision (95% Wilson) | recall | verdict |
|---|---|---|---|---|---|---|---|
| `P_ASYNC` | 14 | 0 | 0 | 36 | 1.000 [0.78, 1.00] | 1.000 | keep |
| `P_UNORD` | 3 | 0 | 4 | 43 | 1.000 [0.44, 1.00] | 0.429 | keep |
| `P_CLOCK` | 12 | 0 | 0 | 38 | 1.000 [0.76, 1.00] | 1.000 | keep |

### Semantic — does it hold in the code's actual behaviour?

| property | precision | recall |
|---|---|---|
| `P_ASYNC` | 0.357 (5/14) | 1.000 |
| `P_UNORD` | 0.667 (2/3) | 0.333 |
| `P_CLOCK` | 0.750 (9/12) | 1.000 |

The two diverge wherever the plan's fixed vocabulary matches a name that is
not doing what the name suggests: `execute` and `start` are listed as async
dispatches, but Hystrix's `cmd.execute()` is the **synchronous** API and
`new SystemCtl(t).start("docker")` merely builds a command object. This is a
property of the definitions, not a defect in the implementation, and it
bounds how much a probe trained on these labels can be said to have learned
about genuine asynchrony.

### Bugs the audit found

An earlier pass over a different 50-method sample surfaced four analyser bugs (timed Thread.join not treated as a barrier; verify(mock).start() counted as a dispatch; map.get(k) results carrying collection taint; try-with-resources initialisers never analysed). Those were fixed before this audit was scored. The P_UNORD tightening below was driven by false positives in THIS sample, so the P_UNORD precision reported here is optimistic and is flagged as such.

Each is now a regression test in `tests/test_structural.py`.

