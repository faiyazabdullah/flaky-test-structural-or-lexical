# 07 — Decision and reporting

Evaluates the stopping rule and produces the write-up. Run only after `01`–`06`
have completed and their acceptance checks pass.

## Stopping rule

Fixed in `00_OVERVIEW.md` before any results were seen. Do not adjust
thresholds, swap in a different baseline, or add a metric after inspecting the
numbers.

**S1 — cue-removed probe beats the strong word baseline, cross-project.**
Paired one-sided test over the 5 folds, probe versus `STRONG_WORD_BASELINE` from
`02`. Wilcoxon signed-rank; report the paired difference and effect size
alongside p. Five folds is little power, so report the fold-level differences in
full and let the reader see them. α = 0.05.

**S2 — transfer drop smaller for the probe than for the word baseline.**
`Δ_probe < Δ_bow`, paired across folds.

**S3 — minimal-pair accuracy above 0.5 for the probe.** Binomial, α = 0.05.
Marked inconclusive rather than failed if fewer than 40 usable pairs. Report
`bow` pair accuracy beside it; the contrast is the substantive result, since a
cue-riding model is expected to land below 0.5.

All three hold → proceed to Phase 2. Otherwise stop and write up.

## Corroboration, not part of the rule

These sharpen the interpretation either way, and belong in the write-up
regardless of outcome: the within- vs cross-project gap from `02`; whether
structural properties alone predict flakiness (`04`); whether representations
encode the structural properties independent of the flakiness label (`06`); and
selectivity at the claimed layers.

Note the possible split verdict: the probe may fail S1 while still encoding
`P_ASYNC` well. That means the model represents the structure but the structure
does not predict this dataset's labels — a different and more interesting
finding than a flat negative, and it should be reported as such.

## If the rule fails

The negative write-up is a deliverable, not a fallback. It should contain:

- the within- vs cross-project gap, quantifying the generalisation problem;
- evidence that cross-project performance is attributable to lexical cues, from
  cue removal and the renaming transfer drop;
- the minimal-pair direction — if `bow` scores below 0.5, cue-riding is shown
  directly rather than inferred;
- the recommendation that cue-ablated, project-grouped baselines become standard
  reporting for flaky test classification;
- the released harness, structural analyser, minimal-pair set, and the
  validation audit.

Do not soften a negative into "promising but inconclusive". The stopping rule
exists so the result can be stated plainly.

## Report

`out/07_report.md`, with every number traced to a JSON file in `out/`, and a
figures script producing: layer curve, within-vs-cross bar chart, transfer drop
comparison, minimal-pair distribution.

## Acceptance

- Every number in the report is reproducible from `out/*.json`.
- Fold-level values reported, not just means.
- Stopping rule verdict stated explicitly, per criterion, with the test
  statistic.
- Limitations section names: the deferred inter-procedural property, static
  analyser precision from `04`, minimal-pair count and mining filters, and the
  two-model scope.
