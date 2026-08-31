# 05 — Minimal pairs from fix commits

The converse test. Renaming holds behaviour fixed and varies surface. This holds
surface nearly fixed and varies behaviour.

## Mining

From IDoFT and the fix-commit links in the FlakyCat lineage, collect pairs
`(x_pre, x_post)` — the same test method before and after a developer's
flakiness fix. Label `x_pre` flaky, `x_post` non-flaky.

Record for each pair: project, test name, category, commit SHA, unified diff,
token-level edit distance, and which cue tokens were added or removed.

Filters: the method must exist on both sides; the fix must touch the test rather
than only production code; and pairs whose diff exceeds 40% of tokens are
excluded as no longer minimal. Persist the excluded ones with reasons — an
honest count of what did not qualify is part of the result.

Target 60–120 usable pairs. Below ~40 the binomial test has too little power and
S3 should be marked inconclusive rather than passed or failed.

## The prediction is signed, not two-tailed

Read this before designing the metric.

Fixes for async flakiness typically **add** `await`, `join()`, or a latch. So
the post-fix version frequently contains *more* cue tokens than the pre-fix one.
The assumption that surface stays constant fails precisely where the data is
densest.

This is more informative than the original design. It splits the hypotheses:

- A model reading cue vocabulary sees the fixed version as *more* flaky and
  scores **below 0.5**.
- A model reading structure scores **above 0.5**.
- Chance is not the interesting comparison. The sign is.

Report pair accuracy for the probe **and** for `bow`, and the contrast between
them is the result. Stratify by whether the fix added cue tokens, removed them,
or was cue-neutral — the cue-neutral subset is the cleanest test and should be
reported separately even if small.

## Metric

For a scorer `s`, over pairs `i`:

```
PairAcc(s) = (1/N) · Σ_i 1[ s(x_pre_i) > s(x_post_i) ]
```

Two-sided binomial test against 0.5. Report a 95% Wilson interval. Also report
the mean margin `s(x_pre) − s(x_post)`, which detects the case where direction
is right but the model is barely separating the pair.

Scorers are trained on the main dataset with **all pair projects held out**. A
pair whose project appeared in training is not a test of generalisation.

## Output

`data/minimal_pairs.csv` with the fields above.
`out/05_pairs.json`: PairAcc, Wilson interval, binomial p, mean margin — per
scorer, overall and per stratum.

## Acceptance

- Every pair's projects are absent from the corresponding training folds. Assert
  this in code; it is the easiest thing here to get silently wrong.
- Cue-token deltas are recorded per pair, so the stratification is possible.
- Excluded pairs are logged with reasons and counted in the write-up.
