# 06 — Representations and linear probes

Source: `out/06_probe.json`. Minimal probing only — no nullspace projection, no
attention analysis; those are Phase 2 and are deliberately absent from this
codebase. Cross-project regime throughout.

## Hardware constraints (Turing, sm_75)

- GPU: NVIDIA GeForce RTX 2070 (sm_75, 7.6 GB), torch 2.6.0+cu124
- **fp16 only** — Turing has no bf16; any `torch.bfloat16` is a bug
- attention implementation is SDPA; FlashAttention-2 requires Ampere
- no 4-bit quantisation: NF4 perturbs the representations being measured

## Extraction

| model / variant | layers | d | max_length | truncation rate | peak VRAM (GB) |
|---|---|---|---|---|---|
| `codebert / code` | 13 | 768 | 512 | 0.1129 | 0.24 |
| `codebert / code_renamed` | 13 | 768 | 512 | 0.1134 | 0.24 |
| `qwen1_5b / code` | 29 | 1536 | 1024 | 0.0085 | 2.88 |
| `qwen1_5b / code_renamed` | 29 | 1536 | 1024 | 0.0087 | 2.88 |

CodeBERT caps at 512 tokens against Qwen's 1024, so the two models see different
amounts of code. That difference is real and is carried into the limitations.

## Probe targets

1. `flaky_code` — flaky on `code`, the headline.
2. `flaky_code` evaluated on `code_renamed` — the transfer drop Δ.
3. `flaky_renamed` — trained *and* evaluated on `code_renamed` (secondary framing).
4. `P_ASYNC` / `P_UNORD` / `P_CLOCK` — does the representation encode the named
   properties at all, with the flakiness label unused?
5. `control` — random labels matched to the class prior, fixed across folds.
   Selectivity S = AP(real) − AP(control). Control prior 0.0325, class prior 0.0325.

## `codebert` / mean pooling — 13 layers, d=768

| target | max AP | at layer | selectivity there | last-layer AP |
|---|---|---|---|---|
| `flaky_code` | 0.4042 ± 0.056 | 5 of 13 | 0.3601 | 0.3649 |
| `flaky_renamed` | 0.4344 ± 0.065 | 5 of 13 | 0.3903 | 0.3463 |
| `control` | 0.0457 ± 0.012 | 0 of 13 | — | 0.0397 |
| `P_ASYNC` | 0.2022 ± 0.085 | 8 of 13 | 0.1672 | 0.1745 |
| `P_UNORD` | 0.1322 ± 0.093 | 8 of 13 | 0.0973 | 0.0871 |
| `P_CLOCK` | 0.2594 ± 0.086 | 10 of 13 | 0.2201 | 0.2528 |

Maxima are quoted with the number of layers searched: they are maxima over a
searched curve, not held-out estimates.

### Full layer curve — `flaky_code`

| layer | AP (mean ± std) | AP on `code_renamed` | Δ | selectivity |
|---|---|---|---|---|
| 0 | 0.1750 ± 0.045 | 0.1580 | +0.0170 | 0.1293 |
| 1 | 0.2772 ± 0.106 | 0.2586 | +0.0186 | 0.2373 |
| 2 | 0.3684 ± 0.084 | 0.3472 | +0.0212 | 0.3273 |
| 3 | 0.3799 ± 0.123 | 0.3683 | +0.0116 | 0.3381 |
| 4 | 0.3761 ± 0.048 | 0.3892 | -0.0130 | 0.3330 |
| 5 | 0.4042 ± 0.056 | 0.4076 | -0.0034 | 0.3601 |
| 6 | 0.3689 ± 0.072 | 0.3516 | +0.0173 | 0.3264 |
| 7 | 0.3472 ± 0.083 | 0.3446 | +0.0027 | 0.3057 |
| 8 | 0.3985 ± 0.078 | 0.4074 | -0.0089 | 0.3635 |
| 9 | 0.3976 ± 0.066 | 0.3992 | -0.0016 | 0.3632 |
| 10 | 0.3586 ± 0.022 | 0.3487 | +0.0099 | 0.3194 |
| 11 | 0.3861 ± 0.080 | 0.3862 | -0.0001 | 0.3512 |
| 12 | 0.3649 ± 0.043 | 0.3465 | +0.0185 | 0.3252 |

## `qwen1_5b` / mean pooling — 29 layers, d=1536

| target | max AP | at layer | selectivity there | last-layer AP |
|---|---|---|---|---|
| `flaky_code` | 0.4234 ± 0.059 | 18 of 29 | 0.3843 | 0.2335 |
| `flaky_renamed` | 0.4204 ± 0.029 | 18 of 29 | 0.3814 | 0.2577 |
| `control` | 0.0494 ± 0.015 | 28 of 29 | — | 0.0494 |
| `P_ASYNC` | 0.2223 ± 0.067 | 12 of 29 | 0.1834 | 0.1268 |
| `P_UNORD` | 0.2440 ± 0.094 | 5 of 29 | 0.2048 | 0.0589 |
| `P_CLOCK` | 0.3757 ± 0.206 | 17 of 29 | 0.3311 | 0.2252 |

Maxima are quoted with the number of layers searched: they are maxima over a
searched curve, not held-out estimates.

### Full layer curve — `flaky_code`

| layer | AP (mean ± std) | AP on `code_renamed` | Δ | selectivity |
|---|---|---|---|---|
| 0 | 0.2540 ± 0.079 | 0.2237 | +0.0303 | 0.2121 |
| 1 | 0.2227 ± 0.049 | 0.1994 | +0.0233 | 0.1846 |
| 2 | 0.3309 ± 0.093 | 0.3112 | +0.0197 | 0.2908 |
| 3 | 0.3587 ± 0.108 | 0.3305 | +0.0282 | 0.3195 |
| 4 | 0.3747 ± 0.104 | 0.3638 | +0.0110 | 0.3354 |
| 5 | 0.3332 ± 0.137 | 0.3184 | +0.0148 | 0.2940 |
| 6 | 0.3864 ± 0.188 | 0.3763 | +0.0101 | 0.3472 |
| 7 | 0.3013 ± 0.132 | 0.2911 | +0.0102 | 0.2661 |
| 8 | 0.2772 ± 0.096 | 0.2614 | +0.0157 | 0.2346 |
| 9 | 0.2876 ± 0.096 | 0.2957 | -0.0082 | 0.2411 |
| 10 | 0.3133 ± 0.081 | 0.3194 | -0.0061 | 0.2734 |
| 11 | 0.2732 ± 0.059 | 0.2694 | +0.0037 | 0.2344 |
| 12 | 0.2947 ± 0.087 | 0.2897 | +0.0049 | 0.2558 |
| 13 | 0.3489 ± 0.111 | 0.3543 | -0.0053 | 0.3105 |
| 14 | 0.3641 ± 0.072 | 0.3643 | -0.0002 | 0.3271 |
| 15 | 0.3647 ± 0.121 | 0.3649 | -0.0002 | 0.3268 |
| 16 | 0.3375 ± 0.087 | 0.3391 | -0.0016 | 0.2986 |
| 17 | 0.4154 ± 0.115 | 0.4026 | +0.0128 | 0.3708 |
| 18 | 0.4234 ± 0.059 | 0.4285 | -0.0051 | 0.3843 |
| 19 | 0.3952 ± 0.037 | 0.3871 | +0.0081 | 0.3560 |
| 20 | 0.3932 ± 0.090 | 0.4009 | -0.0078 | 0.3481 |
| 21 | 0.3824 ± 0.023 | 0.3831 | -0.0008 | 0.3382 |
| 22 | 0.3446 ± 0.093 | 0.3461 | -0.0015 | 0.3025 |
| 23 | 0.3629 ± 0.043 | 0.3740 | -0.0110 | 0.3235 |
| 24 | 0.3226 ± 0.053 | 0.3267 | -0.0041 | 0.2794 |
| 25 | 0.2918 ± 0.102 | 0.3078 | -0.0160 | 0.2491 |
| 26 | 0.2489 ± 0.098 | 0.2596 | -0.0107 | 0.2090 |
| 27 | 0.2956 ± 0.157 | 0.3003 | -0.0047 | 0.2502 |
| 28 | 0.2335 ± 0.093 | 0.2460 | -0.0125 | 0.1841 |

## `qwen1_5b` / last pooling — 29 layers, d=1536

| target | max AP | at layer | selectivity there | last-layer AP |
|---|---|---|---|---|
| `flaky_code` | 0.7085 ± 0.127 | 6 of 29 | 0.6703 | 0.5030 |
| `flaky_renamed` | 0.7125 ± 0.131 | 7 of 29 | 0.6730 | 0.5418 |
| `control` | 0.0452 ± 0.025 | 20 of 29 | — | 0.0379 |
| `P_ASYNC` | 0.2070 ± 0.082 | 10 of 29 | 0.1707 | 0.1316 |
| `P_UNORD` | 0.1565 ± 0.072 | 10 of 29 | 0.1202 | 0.0716 |
| `P_CLOCK` | 0.2104 ± 0.109 | 19 of 29 | 0.1656 | 0.0932 |

Maxima are quoted with the number of layers searched: they are maxima over a
searched curve, not held-out estimates.

### Full layer curve — `flaky_code`

| layer | AP (mean ± std) | AP on `code_renamed` | Δ | selectivity |
|---|---|---|---|---|
| 0 | 0.0351 ± 0.007 | 0.0351 | -0.0000 | -0.0019 |
| 1 | 0.3109 ± 0.101 | 0.3009 | +0.0100 | 0.2714 |
| 2 | 0.5202 ± 0.115 | 0.4813 | +0.0389 | 0.4757 |
| 3 | 0.5700 ± 0.132 | 0.5410 | +0.0290 | 0.5256 |
| 4 | 0.6294 ± 0.100 | 0.5948 | +0.0346 | 0.5855 |
| 5 | 0.6165 ± 0.116 | 0.6035 | +0.0131 | 0.5794 |
| 6 | 0.7085 ± 0.127 | 0.7031 | +0.0053 | 0.6703 |
| 7 | 0.7063 ± 0.136 | 0.7014 | +0.0048 | 0.6668 |
| 8 | 0.6579 ± 0.127 | 0.6578 | +0.0001 | 0.6172 |
| 9 | 0.6343 ± 0.129 | 0.6221 | +0.0122 | 0.6007 |
| 10 | 0.5859 ± 0.103 | 0.5809 | +0.0049 | 0.5495 |
| 11 | 0.5756 ± 0.065 | 0.5692 | +0.0064 | 0.5368 |
| 12 | 0.5613 ± 0.043 | 0.5651 | -0.0038 | 0.5234 |
| 13 | 0.5566 ± 0.069 | 0.5506 | +0.0060 | 0.5182 |
| 14 | 0.5447 ± 0.085 | 0.5212 | +0.0235 | 0.5032 |
| 15 | 0.5488 ± 0.052 | 0.5337 | +0.0151 | 0.5146 |
| 16 | 0.5638 ± 0.060 | 0.5407 | +0.0231 | 0.5238 |
| 17 | 0.5537 ± 0.094 | 0.5429 | +0.0108 | 0.5106 |
| 18 | 0.6257 ± 0.063 | 0.6111 | +0.0145 | 0.5836 |
| 19 | 0.6124 ± 0.066 | 0.5983 | +0.0140 | 0.5676 |
| 20 | 0.6092 ± 0.052 | 0.5998 | +0.0094 | 0.5641 |
| 21 | 0.5663 ± 0.065 | 0.5630 | +0.0033 | 0.5248 |
| 22 | 0.5215 ± 0.078 | 0.5290 | -0.0076 | 0.4771 |
| 23 | 0.5381 ± 0.116 | 0.5379 | +0.0002 | 0.4942 |
| 24 | 0.5366 ± 0.119 | 0.5392 | -0.0026 | 0.4979 |
| 25 | 0.5615 ± 0.132 | 0.5757 | -0.0142 | 0.5233 |
| 26 | 0.5442 ± 0.147 | 0.5513 | -0.0071 | 0.5055 |
| 27 | 0.5189 ± 0.133 | 0.5328 | -0.0139 | 0.4792 |
| 28 | 0.5030 ± 0.129 | 0.5147 | -0.0116 | 0.4651 |

## Acceptance

| check | result |
|---|---|
| peak VRAM under 7.5 GB | 2.88 GB — PASS |
| selectivity positive at the claimed layer (`codebert|mean`) | PASS |
| selectivity positive at the claimed layer (`qwen1_5b|mean`) | PASS |
| selectivity positive at the claimed layer (`qwen1_5b|last`) | PASS |
| CodeBERT truncation above 20% | no |
