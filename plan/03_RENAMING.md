# 03 — Cue removal and counterfactual renaming

Two interventions on the lexical channel. Both produce variants of `code` and a
shared cue vocabulary used by `02` and `06`.

## Cue vocabulary

Assemble `data/cue_vocab.json` from three sources, keeping them separately
tagged so each can be ablated alone:

- **attributed**: tokens reported as top attributions by Rahman et al.
  (OOPSLA'25) — `sleep`, `wait`, `Interrupted`, and the per-category lists.
- **api**: nondeterminism-related API and type names — `Thread`, `Thread.sleep`,
  `await*`, `Awaitility`, `CountDownLatch`, `Semaphore`, `ExecutorService`,
  `Future`, `CompletableFuture`, `synchronized`, `volatile`, `Concurrent*`,
  `HashMap`, `HashSet`, `LinkedHashMap`, `keySet`, `entrySet`, `values`,
  `Random`, `currentTimeMillis`, `nanoTime`, `TimeUnit`, `Date`, `Calendar`,
  `LocalDateTime`, `Instant`, `now`.
- **mined**: top-k identifiers by mutual information with the label, computed on
  training folds only. Never on the full set — that leaks.

## Intervention A — removal

Drop cue tokens from the vectoriser vocabulary. Feature-space only; the code is
untouched. Feeds `bow_ablated` in `02`.

## Intervention B — counterfactual renaming

Rewrite the source. Replace each cue with a neutral identifier of matched shape:
`Thread.sleep` → `Util.pause`, `CountDownLatch` → `Gate`, `HashMap` → `MapA`,
`currentTimeMillis` → `readCounter`. Preserve control flow, call structure,
arity, and assertions. Write as column `code_renamed`.

Requirements:

- Word-boundary matching. `sleepTimeout` must not become `pauseTimeout` unless
  it is genuinely the same identifier.
- Renaming is a bijection with a persisted map, so it can be inverted and audited.
- Do not rename inside string literals or comments. Strip comments first, or
  parse — do not regex over raw source and hope.
- **Unit tests required.** At minimum: cue inside a string literal is untouched;
  substring of a longer identifier is untouched; the same cue appearing twice
  maps consistently; token count is unchanged.

## Transfer evaluation

Train on `code`, evaluate on `code_renamed`. Report

```
Δ = AP(model; code) − AP(model; code_renamed)
```

under the cross-project regime, for every method in `02` and later for the probe
in `06`.

**Retraining on `code_renamed` measures nothing** and must not be reported as
the headline. A retrained bag-of-tokens model simply relearns `pause` and `Gate`
and scores identically. The asymmetry between train and eval distributions is
the whole measurement. Compute the retrained number once as a sanity check that
it lands within noise of the original, then set it aside.

## Output

`data/prepped.csv` gains `code_renamed`. `data/cue_vocab.json`,
`data/rename_map.json`, `out/03_transfer.json` with Δ per method.

## Acceptance

- All rename unit tests pass.
- Token count of `code_renamed` equals that of `code` for at least 95% of rows;
  investigate the remainder rather than ignoring it.
- Retrained-on-renamed AP is within one fold-std of AP on `code`. If it is not,
  the renaming is not shape-preserving and is leaking information.
