# 06 — Representations and probe

Minimal probing only. No nullspace projection, no attention analysis — those are
Phase 2 and must not appear in this codebase.

## Models

| model | params | fp16 VRAM | role |
|-------|--------|-----------|------|
| `microsoft/codebert-base` | 125M | ~0.3GB | encoder reference; backs FlakyCat |
| `Qwen/Qwen2.5-Coder-1.5B` | 1.5B | ~3.1GB | decoder |

`Qwen/Qwen2.5-Coder-3B` (~6.2GB fp16) only at batch size 1, and only after the
1.5B pipeline is validated end to end.

**Turing constraints — violating these produces silent garbage or a crash:**

- `torch_dtype=torch.float16`. Not bfloat16. sm_75 has no bf16 support.
- `attn_implementation="sdpa"` or `"eager"`. FlashAttention-2 requires Ampere.
- No 4-bit quantisation in Phase 1.
- `max_length=1024`; log the truncation rate. CodeBERT caps at 512 — a separate
  cap, and the difference must be reported since the two models then see
  different amounts of code.

## Extraction

One forward pass per test, `output_hidden_states=True`. Pool by mean over code
tokens, excluding padding and special tokens. For the decoder, also store the
final-position state as a secondary representation.

Save `out/emb_{model}_{variant}_{pool}.npy`, shape `[n, L+1, d]`, fp16, row
order matching `data/prepped.csv`. Variants: `code`, `code_renamed`.

For batching, sort by length and restore original order afterwards. Use
`padding_side='left'` for the decoder so the final position is a real token.
Under 9k samples at these sizes this runs in well under an hour on a 2070.

## Probes

Linear only: `LogisticRegression(class_weight='balanced')`, `C` tuned on an
inner split. Report AP per layer under the cross-project regime. Report the
full layer curve, never a selected best layer — and if a maximum is quoted,
quote the number of layers searched alongside it.

Four probe targets:

1. `flaky` on `code` — the headline.
2. `flaky` on `code`, evaluated on `code_renamed` — transfer drop Δ (see `03`).
3. `P_ASYNC`, `P_UNORD`, `P_CLOCK` separately — does the representation encode
   the named properties at all, with the flakiness label unused.
4. Control task: random labels matched to the class prior, fixed across folds.
   Report selectivity `S = AP(real) − AP(control)`. A probe with high AP and
   near-zero selectivity has capacity, not information.

## Output

`out/emb_*.npy`, `out/06_probe.json` with per-layer AP mean and std, fold-level
vectors, selectivity, transfer drop, and per-property AP.

## Acceptance

- Embedding row count and order match `prepped.csv` exactly. Assert on ids.
- Truncation rate reported per model; if above 20% for CodeBERT at 512, say so
  explicitly in the write-up rather than burying it.
- Selectivity is positive at the layers being claimed.
- Peak VRAM logged and under 7.5GB.
