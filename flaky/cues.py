"""Cue vocabulary: the lexical channel we intervene on.

Three separately-tagged sources so each can be ablated alone:

``attributed``
    Tokens the plan cites from Rahman et al. (OOPSLA'25) as top attributions.
    Only the three the plan names verbatim (``sleep``, ``wait``,
    ``Interrupted``) carry that provenance; the per-category extension is
    curated here and tagged ``attributed_curated`` so the distinction survives
    into ``data/cue_vocab.json``.  Do not report the curated list as the
    paper's.

``api``
    Nondeterminism-related API and type names, from the plan's list.

``mined``
    Top-k identifiers by mutual information with the label.  Computed on
    *training folds only* -- computing it on the full set leaks.  The copy
    written to ``data/cue_vocab.json`` is fitted on the whole dataset and is
    audit-only; it is never used to fit anything.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, Sequence

from .javalex import identifiers

# --------------------------------------------------------------------------
# static sources
# --------------------------------------------------------------------------

ATTRIBUTED = [
    "sleep",
    "wait",
    "Interrupted",
    "InterruptedException",
]

# Per-category extension.  Curated here, NOT taken from the paper.
ATTRIBUTED_CURATED = {
    "async_wait": ["await", "Awaitility", "timeout", "poll", "eventually", "assertEventually",
                   "retry", "waitFor", "waitUntil", "delay"],
    "concurrency": ["Thread", "thread", "concurrent", "Concurrent", "lock", "Lock", "atomic",
                    "Atomic", "race", "parallel", "Parallel"],
    "time": ["time", "Time", "clock", "Clock", "timestamp", "millis", "nanos", "Duration",
             "elapsed"],
    "unordered_collections": ["Hash", "hash", "unordered", "order", "sorted", "Sorted", "shuffle",
                              "iterator", "Iterator"],
    "test_order_dependency": ["static", "setUp", "tearDown", "BeforeClass", "AfterClass",
                              "FixMethodOrder", "MethodSorters", "shared", "reset", "clear"],
}

# Nondeterminism-related API and type names.  ``*`` is a prefix wildcard,
# expanded against the identifiers actually present in the corpus.
API = [
    "Thread", "Thread.sleep", "await*", "Awaitility", "CountDownLatch", "Semaphore",
    "ExecutorService", "Executors", "Executor", "Future", "CompletableFuture", "synchronized",
    "volatile", "Concurrent*", "HashMap", "HashSet", "LinkedHashMap", "LinkedHashSet", "keySet",
    "entrySet", "values", "Random", "currentTimeMillis", "nanoTime", "TimeUnit", "Date",
    "Calendar", "LocalDateTime", "Instant", "now", "join", "submit", "execute", "invokeAll",
    "schedule", "ScheduledExecutorService", "AtomicInteger", "AtomicLong", "AtomicBoolean",
    "AtomicReference", "ReentrantLock", "CyclicBarrier", "Phaser", "ForkJoinPool", "sleep",
    "notify", "notifyAll", "System.currentTimeMillis", "System.nanoTime", "Instant.now",
    "listFiles", "toArray", "Runnable", "Callable", "runAsync", "supplyAsync", "thenApply",
]


def _expand(entry: str, corpus_idents: set[str]) -> set[str]:
    """Turn a vocabulary entry into the set of *identifier tokens* it covers.

    Renaming happens at identifier granularity, which is what makes the token
    count exactly preservable.  ``Thread.sleep`` therefore contributes the two
    identifiers ``Thread`` and ``sleep``; ``await*`` contributes every corpus
    identifier starting with ``await``.
    """
    out: set[str] = set()
    if entry.endswith("*"):
        pref = entry[:-1]
        out |= {i for i in corpus_idents if i.startswith(pref) and i != pref} | {pref}
        return {o for o in out if o}
    for part in entry.split("."):
        part = part.strip()
        if part:
            out.add(part)
    return out


def build_static_vocab(corpus_idents: Iterable[str]) -> dict[str, list[str]]:
    """The parts of the cue vocabulary that do not touch the label."""
    idents = set(corpus_idents)
    tags: dict[str, set[str]] = {
        "attributed": set(),
        "attributed_curated": set(),
        "api": set(),
    }
    for e in ATTRIBUTED:
        tags["attributed"] |= _expand(e, idents)
    for cat, lst in ATTRIBUTED_CURATED.items():
        for e in lst:
            tags["attributed_curated"] |= _expand(e, idents)
    for e in API:
        tags["api"] |= _expand(e, idents)

    # Keep the tags disjoint in priority order attributed > api > curated, so a
    # per-tag ablation removes a well-defined set.
    tags["api"] -= tags["attributed"]
    tags["attributed_curated"] -= tags["attributed"] | tags["api"]
    return {k: sorted(v) for k, v in tags.items()}


# --------------------------------------------------------------------------
# mined cues (training folds only)
# --------------------------------------------------------------------------


def mine_cues(codes: Sequence[str], labels: Sequence[int], top_k: int = 200,
              min_df: int = 3) -> list[str]:
    """Top-k identifiers by mutual information with the binary label.

    ``codes``/``labels`` must be the *training* rows of a fold.  MI is computed
    on the presence/absence indicator per document, which is what a TF-IDF
    identifier model keys on.
    """
    import numpy as np

    doc_idents = [set(identifiers(c)) for c in codes]
    df = Counter()
    for s in doc_idents:
        df.update(s)
    vocab = sorted(t for t, c in df.items() if c >= min_df)
    if not vocab:
        return []
    index = {t: i for i, t in enumerate(vocab)}

    y = np.asarray(labels, dtype=np.int64)
    n = len(y)
    n1 = int(y.sum())
    n0 = n - n1
    if n1 == 0 or n0 == 0:
        return []

    # counts of (token present, y=1) and (token present, y=0)
    c_t1 = np.zeros(len(vocab), dtype=np.float64)
    c_t0 = np.zeros(len(vocab), dtype=np.float64)
    for s, yi in zip(doc_idents, y):
        tgt = c_t1 if yi else c_t0
        for t in s:
            j = index.get(t)
            if j is not None:
                tgt[j] += 1.0

    eps = 1e-12
    p_t1, p_t0 = c_t1 / n, c_t0 / n
    p_f1, p_f0 = (n1 - c_t1) / n, (n0 - c_t0) / n
    p_t = p_t1 + p_t0
    p_f = 1.0 - p_t
    py1, py0 = n1 / n, n0 / n

    mi = (
        p_t1 * np.log((p_t1 + eps) / (p_t * py1 + eps))
        + p_t0 * np.log((p_t0 + eps) / (p_t * py0 + eps))
        + p_f1 * np.log((p_f1 + eps) / (p_f * py1 + eps))
        + p_f0 * np.log((p_f0 + eps) / (p_f * py0 + eps))
    )
    order = np.argsort(-mi)
    return [vocab[i] for i in order[:top_k]]


# --------------------------------------------------------------------------
# neutral replacements for the counterfactual renaming (intervention B)
# --------------------------------------------------------------------------

# Curated, shape-matched replacements for the cues that matter most.  Anything
# not listed gets a deterministic generated name of the same casing shape.
CURATED_RENAMES = {
    "Thread": "Util",
    "sleep": "pause",
    "wait": "hold",
    "notify": "signal",
    "notifyAll": "signalAll",
    "Interrupted": "Halted",
    "InterruptedException": "HaltedFailure",
    "CountDownLatch": "Gate",
    "Semaphore": "Permit",
    "CyclicBarrier": "Ring",
    "Phaser": "Stager",
    "ReentrantLock": "Guard",
    "ExecutorService": "Runner",
    "ScheduledExecutorService": "TimedRunner",
    "Executors": "Runners",
    "Executor": "Dispatcher",
    "ForkJoinPool": "SplitPool",
    "Future": "Handle",
    "CompletableFuture": "ChainHandle",
    "runAsync": "startChain",
    "supplyAsync": "produceChain",
    "thenApply": "andMap",
    "submit": "post",
    "execute": "invokeOn",
    "invokeAll": "postAll",
    "schedule": "postLater",
    "join": "settle",
    "await": "stall",
    "Awaitility": "Stallity",
    "HashMap": "MapA",
    "LinkedHashMap": "MapB",
    "HashSet": "SetA",
    "LinkedHashSet": "SetB",
    "keySet": "firstView",
    "entrySet": "pairView",
    "values": "secondView",
    "toArray": "toBlock",
    "listFiles": "listEntries",
    "currentTimeMillis": "readCounter",
    "nanoTime": "readFineCounter",
    "now": "current",
    "Instant": "Mark",
    "LocalDateTime": "MarkLocal",
    "Date": "Stamp",
    "Calendar": "Almanac",
    "TimeUnit": "Scale",
    "Duration": "Span",
    "Random": "Picker",
    "AtomicInteger": "CellInt",
    "AtomicLong": "CellLong",
    "AtomicBoolean": "CellBool",
    "AtomicReference": "CellRef",
    "Runnable": "Job",
    "Callable": "Producer",
    "volatile": "volatile",      # a keyword: never renamed, listed for clarity
    "synchronized": "synchronized",
}

_UPPER_SNAKE = re.compile(r"^[A-Z][A-Z0-9_$]*$")
_UPPER_CAMEL = re.compile(r"^[A-Z]")


def _shape_name(name: str, idx: int) -> str:
    """Deterministic neutral name preserving the casing convention of ``name``."""
    suffix = _base36(idx)
    if _UPPER_SNAKE.match(name) and "_" in name:
        return f"CUE_{suffix.upper()}"
    if _UPPER_SNAKE.match(name) and len(name) <= 3:
        return f"CUE{suffix.upper()}"
    if _UPPER_CAMEL.match(name):
        return f"Sym{suffix.capitalize()}"
    return f"sym{suffix.capitalize()}"


def _base36(n: int) -> str:
    digits = "abcdefghijklmnopqrstuvwxyz0123456789"
    if n == 0:
        return "a"
    out = []
    while n:
        n, r = divmod(n, 36)
        out.append(digits[r])
    return "".join(reversed(out))


def build_rename_map(cue_idents: Sequence[str], corpus_idents: set[str]) -> dict[str, str]:
    """A bijection cue-identifier -> neutral identifier.

    Guarantees, asserted by the caller and by ``tests/test_rename.py``:
    injective, and no replacement collides with an identifier already present
    in the corpus (which would silently merge two distinct symbols).
    """
    from .javalex import JAVA_KEYWORDS

    taken = set(corpus_idents)
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for name in sorted(cue_idents):
        if name in JAVA_KEYWORDS:
            continue  # keywords are not identifiers; renaming them breaks the code
        cand = CURATED_RENAMES.get(name)
        if cand is None or cand == name or cand in taken or cand in used:
            k = 0
            while True:
                cand = _shape_name(name, len(mapping) + k)
                if cand not in taken and cand not in used and cand != name:
                    break
                k += 1
        mapping[name] = cand
        used.add(cand)
    assert len(set(mapping.values())) == len(mapping), "rename map is not injective"
    return mapping
