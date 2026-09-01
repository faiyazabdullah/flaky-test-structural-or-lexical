# Structural or lexical? Probing a code LM for test flakiness

Implementation of the study specified in [`plan/`](plan/). One question:

> After removing lexical cues, does any predictive signal for flakiness survive
> across unseen projects — and is that signal attributable to named structural
> properties rather than to project or dataset artefacts?

The answer is decided by a **stopping rule fixed in advance** (`plan/00_OVERVIEW.md`)
and evaluated in `scripts/s07_decision.py`. A negative result is a deliverable,
not a fallback.

Probing with nullspace projection and attention analysis are **out of scope** for
Phase 1 and are deliberately absent from this codebase.

## Install

```bash
python3 -m venv .venv
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
.venv/bin/python -m pip install -r requirements.txt
```

Torch comes from the CUDA 12.4 index because the target card is an RTX 2070
(Turing, sm_75). **fp16 only** — Turing has no bf16, FlashAttention-2 needs
Ampere, and 4-bit quantisation is banned in Phase 1 because NF4 perturbs the
representations being measured. `flaky/embed.py:assert_turing_safe` fails loudly
rather than letting any of those produce silent garbage.

## Run

```bash
.venv/bin/python scripts/run_all.py --smoke   # ~12 min, validates the pipeline end to end
.venv/bin/python scripts/run_all.py           # the real run
```

Smoke mode is fully isolated: it reads and writes `data_smoke/`, `out_smoke/`,
`ckpt_smoke/`, so it can never contaminate real results. Its acceptance failures
are reported but non-fatal — a few hundred rows cannot satisfy the
sample-size-dependent criteria, and the point of the smoke run is plumbing.

The 3B stretch target is available but off by default — run it only after the
1.5B pipeline has completed end to end, and only at batch size 1:

```bash
.venv/bin/python scripts/s06_probe.py --models qwen3b --batch-size 1
```

Everything is resumable. Each stage writes `ckpt/<stage>.done` recording the
inputs it saw and skips itself if they are unchanged; long loops inside a stage
append to a JSONL checkpoint after every item, fsynced, so a hard kill loses at
most the item in flight. Kill any stage and re-run the same command.

```bash
.venv/bin/python scripts/run_all.py --from 06        # resume at a stage
.venv/bin/python scripts/run_all.py --only 02,07     # re-run specific stages
.venv/bin/python scripts/run_all.py --force          # ignore all checkpoints
```

Progress bars go to stdout; every stage also appends to `logs/<stage>.log`.

## Stage order

The plan numbers files `01 → 07`; the real dependency order differs in one
place, and `run_all.py` follows the dependencies:

| order | stage | what it produces |
|---|---|---|
| 1 | `s01_prep.py` | `data/prepped.csv`, `out/01_prep.json` |
| 2 | `s03_cues_rename.py` | `data/cue_vocab.json`, `data/rename_map.json`, `code_renamed` |
| 3 | `s02_baselines.py` | `out/02_baselines.json`, `out/03_transfer.json` |
| 4 | `s04_structural.py` | `data/structural.csv`, `out/04_structural.json`, audit sample |
| 5 | `s05_mine_pairs.py` | `data/minimal_pairs.csv` (+ the excluded set, with reasons) |
| 6 | `s06_probe.py` | `out/emb_*.npy`, `out/06_probe.json` |
| 7 | `s04_validate.py` | `out/04_validation.md` |
| 8 | `s05_score_pairs.py` | `out/05_pairs.json` |
| 9 | `s07_decision.py` | `out/07_decision.json`, `out/07_report.md` |
| 10 | `scripts/figures.py` | `out/figures/*.png` |
| 11 | `scripts/export_reports.py` | `reports/*.md` — every result as self-contained Markdown |

`03` runs before `02` because `bow_ablated` *is* `bow` with `03`'s cue
vocabulary removed, and because the renaming transfer drop reuses the models
`02` has already fitted. `05` splits: mining needs only the corpus, scoring
needs the probe from `06`.

## Reading the results

`out/` holds the machine-readable artefacts (JSON per stage, `.npy` embeddings,
PNG figures). `scripts/export_reports.py` turns them into a directory of
self-contained Markdown:

```bash
.venv/bin/python scripts/export_reports.py     # -> reports/
```

Start at `reports/00_INDEX.md`, then `reports/MAIN_REPORT.md`. Every table names
the JSON file it was generated from, so nothing in the prose can drift away from
the artefact behind it.

## The hand audit

The audit is what turns the structural labels into ground truth, and it found
real bugs: a timed `Thread.join(2000)` not treated as a barrier, `verify(mock)
.start()` counted as an async dispatch, `map.get(k)` results still carrying
their map's collection taint, try-with-resources initialisers never analysed at
all, and a P_UNORD rule loose enough to fire whenever a collection was passed to
any method. Each is now a regression test in `tests/test_structural.py`.

One caveat the code cannot fix: both models were pretrained on public GitHub,
which includes these repositories. A probe that beats a word baseline may be
reading memorised project identity rather than program structure. Project-grouped
CV stops the *classifier* seeing a test project's rows; it cannot stop the
*encoder* having seen the repository. `out/07_report.md` states this next to the
S1 verdict.

`out/04_validation.md` scores two judgements per property: the **definitional**
one (does the property hold as the plan writes it, with its fixed
vocabularies) — this is the acceptance gate — and the **semantic** one (does it
hold in the code's actual behaviour). They diverge sharply for `P_ASYNC`,
because `execute` and `start` are listed as async dispatches but Hystrix's
`cmd.execute()` is the synchronous API. That gap bounds what a probe trained on
these labels can be said to have learned, and the report says so.


`s04_structural.py` writes `out/04_audit_sample.md` — 50 methods stratified by
property and by label — and `data/04_audit_template.json`. An auditor reads each
method, judges each property against the definitions in `plan/04`, and saves the
judgements as `data/04_audit_labels.json`. `s04_validate.py` turns those into
precision and recall and enforces the 0.8 precision floor.

**An unvalidated static analyser is not ground truth.** If the audit has not
been run, `out/07_report.md` says so in place of the validation table.

## Layout

```
flaky/            library code
  common.py       paths, logging, checkpoints, acceptance reporting
  normalize.py    formatting normalisation (the whitespace artefact)
  javalex.py      dependency-free Java lexer
  cues.py         cue vocabulary, MI mining, neutral-name generation
  rename.py       counterfactual renaming (intervention B)
  structural.py   tree-sitter taint analysis for P_ASYNC / P_UNORD / P_CLOCK
  javamethods.py  Class.method extraction from whole files (for 05)
  gitfetch.py     partial-clone commit fetching, cached
  cv.py           CV regimes, featurizers, fit/score loop
  embed.py        hidden-state extraction, Turing-safe
  stats.py        Wilson, binomial, paired Wilcoxon
scripts/          one CLI per stage + run_all.py, report.py, figures.py
tests/            unit tests for normalisation, renaming, structural analysis
data/  out/  ckpt/  logs/  cache/
```

`cache/` holds network artefacts (HuggingFace models, git objects) and is shared
between smoke and full runs on purpose — those are inputs, not results.

## Four things worth knowing before reading any number

1. **The released CSV is separable by whitespace alone**, AUROC 1.000. No flaky
   test begins with indentation or ends with a newline; 98.9% and 100% of
   non-flaky tests do. `01` normalises this away and asserts the result;
   without that step every downstream number measures which extraction pipeline
   produced the row.
2. **Six test methods appear twice with identical code and contradictory
   labels.** Both copies are dropped and listed in `out/01_prep.json`; picking a
   winner would be inventing ground truth.
3. **~4% of method bodies are truncated mid-statement** by the FlakeBench
   extraction, almost all of them non-flaky. `04` repairs the brackets or
   rewinds to the last complete statement, flags every repaired row, and reports
   the rate with and without repair.
4. **The `id` column is not unique.** FlakeBench numbers its flaky and
   non-flaky halves independently and 66 ids collide across them, so anything
   keyed on `id` — a checkpoint, a join — silently mixes two different tests.
   `01` derives a content-hash `uid` and everything downstream keys on that.
   `id` is kept for provenance.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

`run_all.py` runs them first and refuses to start the pipeline if they fail.
Renaming and structural analysis are where a silent bug invalidates everything
downstream, which is why those two have tests and the rest have assertions.
