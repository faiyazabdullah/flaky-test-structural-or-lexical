# 05 — Minimal pairs from fix commits

Sources: `out/05_mine.json`, `out/05_pairs.json`, `data/minimal_pairs.csv`,
`data/minimal_pairs_excluded.csv`.

The converse of the renaming test. Renaming holds behaviour fixed and varies
surface; this holds surface nearly fixed and varies behaviour: `x_pre` is a test
method before a developer's flakiness fix, `x_post` the same method after.

## Mining

| | |
|---|---|
| fix commits considered | 172 |
| commits that could not be fetched | 2 |
| **usable pairs** | **232** |
| … matching a FlakeBench flaky test by name | 50 |
| … other test methods the same commits touched | 182 |
| projects | 65 |
| pairs excluded | 47 |

### Why pairs were excluded

| reason | count |
|---|---|
| `diff_exceeds_40pct_of_tokens` | 13 |
| `no_test_file_changed` | 22 |
| `file_missing_on_one_side` | 11 |
| `method_missing_on_one_side` | 1 |

An honest count of what did not qualify is part of the result; the excluded
set is persisted with reasons in `data/minimal_pairs_excluded.csv`.

### Composition

- cue strata: {'cue_added': 41, 'cue_neutral': 162, 'cue_removed': 29}
- annotation-only fixes (`@Ignore` and the like, which change no behaviour): 17
- categories: {'unknown': 182, 'async wait': 25, 'unordered collections': 12, 'time': 6, 'concurrency': 4, 'test order dependency': 3}
- median diff fraction: 0.0508 (pairs above 0.4 were excluded as no longer minimal)

## The prediction is signed, not two-tailed

A cue-reading model scores BELOW 0.5, because fixes usually add cue tokens (await, join, a latch). A structure-reading model scores above it. The contrast between the probe and bow is the result; chance is not the interesting comparison.

```
PairAcc(s) = (1/N) · Σ_i  1[ s(x_pre_i) > s(x_post_i) ]
```

## Two pair sets

Pair count and scorer strength trade against each other: holding out a
pair's project is mandatory, so more pairs means fewer training projects.
Both are reported.

| pair set | pairs | held-out projects | training rows | training flaky |
|---|---|---|---|---|
| `flakebench_matched` *(primary)* | 50 | 36 | 5333 | 191 |
| `all` | 232 | 65 | 2679 | 84 |

### Results — `flakebench_matched` (primary — S3 reads this one)

| scorer | PairAcc | 95% Wilson | binomial p | mean margin |
|---|---|---|---|---|
| `bow` | 0.4800 | [0.348, 0.615] | 0.8877 | +0.0363 |
| `probe_codebert_mean` | 0.2400 | [0.143, 0.374] | 0.0003 | -0.2143 |
| `probe_qwen1_5b_mean` | 0.3800 | [0.259, 0.518] | 0.1189 | -0.2685 |
| `probe_qwen1_5b_last` | 0.4200 | [0.294, 0.558] | 0.3222 | -0.0857 |

By stratum:

| scorer | all | annotation_only | behavioural_only | cue_added | cue_neutral | cue_removed |
|---|---|---|---|---|---|---|
| `bow` | 0.4800 (n=50) | 0.6667 (n=3) | 0.4681 (n=47) | 0.3571 (n=14) | 0.4848 (n=33) | 1.0000 (n=3) |
| `probe_codebert_mean` | 0.2400 (n=50) | 0.3333 (n=3) | 0.2340 (n=47) | 0.2143 (n=14) | 0.2424 (n=33) | 0.3333 (n=3) |
| `probe_qwen1_5b_mean` | 0.3800 (n=50) | 0.6667 (n=3) | 0.3617 (n=47) | 0.1429 (n=14) | 0.4848 (n=33) | 0.3333 (n=3) |
| `probe_qwen1_5b_last` | 0.4200 (n=50) | 1.0000 (n=3) | 0.3830 (n=47) | 0.6429 (n=14) | 0.3333 (n=33) | 0.3333 (n=3) |

### Results — `all`

| scorer | PairAcc | 95% Wilson | binomial p | mean margin |
|---|---|---|---|---|
| `bow` | 0.4224 | [0.361, 0.487] | 0.0214 | -0.0000 |
| `probe_codebert_mean` | 0.4612 | [0.398, 0.525] | 0.2643 | +0.0035 |
| `probe_qwen1_5b_mean` | 0.4483 | [0.386, 0.513] | 0.1309 | -0.2159 |
| `probe_qwen1_5b_last` | 0.4914 | [0.428, 0.555] | 0.8439 | +0.0848 |

By stratum:

| scorer | all | annotation_only | behavioural_only | cue_added | cue_neutral | cue_removed |
|---|---|---|---|---|---|---|
| `bow` | 0.4224 (n=232) | 0.1765 (n=17) | 0.4419 (n=215) | 0.3415 (n=41) | 0.3951 (n=162) | 0.6897 (n=29) |
| `probe_codebert_mean` | 0.4612 (n=232) | 0.4706 (n=17) | 0.4605 (n=215) | 0.4390 (n=41) | 0.4630 (n=162) | 0.4828 (n=29) |
| `probe_qwen1_5b_mean` | 0.4483 (n=232) | 0.4706 (n=17) | 0.4465 (n=215) | 0.5366 (n=41) | 0.4321 (n=162) | 0.4138 (n=29) |
| `probe_qwen1_5b_last` | 0.4914 (n=232) | 0.4118 (n=17) | 0.4977 (n=215) | 0.6341 (n=41) | 0.4568 (n=162) | 0.4828 (n=29) |

The **cue-neutral** subset is the cleanest test: neither side of the pair gains
or loses cue tokens, so a scorer reading vocabulary has nothing to go on.

