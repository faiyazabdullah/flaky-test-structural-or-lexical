# 04 — Structural property labels

Ground truth that does not depend on the flakiness label. This is what makes
"structural information" a testable claim rather than a residue.

## Properties

Intra-procedural, computable from an isolated method body.

- **`P_ASYNC`** — an async dispatch (`submit`, `execute`, `start`,
  `thenApply`, `runAsync`, `supplyAsync`, `schedule`) reaches an assertion with
  no intervening synchronisation (`get`, `join`, `await`, `latch.await`,
  `sleep`, `invokeAll`) on the path.
- **`P_UNORD`** — a value originating from unordered iteration
  (`HashMap`/`HashSet` traversal, `keySet`, `entrySet`, `values`, `listFiles`,
  `toArray` on an unordered collection) reaches an assertion argument.
- **`P_CLOCK`** — a clock read (`currentTimeMillis`, `nanoTime`, `now`,
  `new Date()`, `Instant.now`) reaches an assertion argument.

**Not implemented.** Shared field written by one test and read by another is
inter-procedural, needs the class body and sibling tests, and cannot be computed
from FlakeBench method bodies. Record this as a stated limitation; it moves to
Phase 2 with repository checkout.

## Implementation

`tree-sitter` with `tree-sitter-java`, not regex and not `javalang`. Test
methods are fragments, so wrap each in a synthetic class before parsing:

```java
class Synth { <method body here> }
```

Record the parse failure rate. If it exceeds 5%, inspect before continuing —
a systematically unparseable subset will bias the labels.

Analysis is a taint-style intra-procedural reaching-definitions pass:

1. Build the CFG over statements in the method.
2. Mark source nodes (async dispatch, unordered iteration, clock read).
3. Propagate taint through local variable assignments and method call arguments.
4. Mark sink nodes: arguments of `assert*`, `verify`, `expect*`, `fail`.
5. For `P_ASYNC`, kill taint at a synchronisation call.

Aliasing, field access, and lambda capture are all approximated. That is
acceptable — but the approximation must be measured, not assumed.

## Validation — do not skip

Sample 50 methods stratified by property and by label. Hand-audit each. Report
precision and recall per property in `out/04_validation.md`. If precision for any
property is below 0.8, either fix the analysis or drop the property from the
study. An unvalidated static analyser is not ground truth.

## Two experiments this enables

**Do structural properties predict flakiness?** Fit a logistic regression on the
three booleans alone, cross-project. If they carry real signal, the properties
are meaningful; if not, either the analysis is weak or flakiness in this dataset
is not captured by them. Either way it is reportable.

**Do model representations encode the properties?** In `06`, probe for each
structural property as the target, with the *flakiness* label unused. This is
the cleanest version of the question: does the model represent async-reaching-
assertion at all, independent of whether that helps it predict flakiness.

## Output

`data/structural.csv`: `id, P_ASYNC, P_UNORD, P_CLOCK, parse_ok`.
`out/04_structural.json`: prevalence per property overall and per category,
cross-tab against `label`, parse failure rate.
`out/04_validation.md`: the hand audit.

## Acceptance

- Parse failure rate under 5%, with failures listed.
- Precision ≥ 0.8 per retained property on the audit sample.
- `P_ASYNC` prevalence is higher in the async-wait category than in non-flaky
  tests. If it is not, the analysis is broken — that is a floor check, not a
  finding.
