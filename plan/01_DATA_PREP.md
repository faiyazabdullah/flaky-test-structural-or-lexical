# 01 — Data preparation

Build `data/prepped.csv` from the FlakeBench CSV. Everything downstream assumes
this file's row order.

## Input

FlakeBench CSV with columns `id, project, test_name, full_code, label, category`.
Labels are `non-flaky` plus five root causes: async wait, concurrency, time,
unordered collections, test order dependency.

## Steps

**Deduplicate** on exact `full_code`. Assert first that no code string carries
conflicting labels; if the assertion fires, stop and report rather than picking
a winner.

**Normalise formatting — mandatory, not a preprocessing nicety.** Convert CRLF
and CR to LF, expand tabs to four spaces, strip trailing whitespace per line,
dedent by common leading indentation, strip leading and trailing blank lines.

The reason: in the released CSV the two classes are separable by formatting
alone. No flaky test begins with indentation or ends with a newline; 98.9% and
100% of non-flaky tests do. Four boolean features give AUROC 1.000. The classes
were evidently extracted by different pipelines. A subword tokenizer encodes
leading whitespace, so without this step every result downstream is measuring
provenance. Assert post-normalisation that the leading-indent rate is 0 and that
no code ends in a newline.

**Derive columns**: `flaky` (label != `non-flaky`), `code` (normalised),
`n_chars`, `n_lines`, `n_tokens` (CodeBERT tokenizer length).

**Length-matched subset.** Flaky tests are ~1.7× longer (median 799 vs 470 chars
after normalisation) and length alone reaches AUROC ≈ 0.72. Build a 1:1
nearest-neighbour match on `n_tokens`, caliper 0.05 relative, preferring a
partner within the same project and falling back to global. Flag as `matched`.

## Output

`data/prepped.csv`: `id, project, test_name, code, flaky, label, category,
n_chars, n_lines, n_tokens, matched`.

`out/01_prep.json`: row counts before and after dedup, class counts, project
count, matched subset size, and the post-normalisation whitespace assertions.

## Acceptance

- No duplicate `code` values.
- Leading-indent rate exactly 0.0.
- A logistic regression on the four whitespace booleans scores AUROC ≈ 0.5
  (± 0.05). If it does not, normalisation is incomplete — fix before proceeding.
- Matched subset is exactly balanced, with median `n_tokens` equal across
  classes to within 2%.
