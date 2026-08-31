"""Intervention B -- counterfactual renaming.

Rewrites the source, replacing each lexical cue with a neutral identifier of
matched shape while preserving control flow, call structure, arity and
assertions.  Operates on the lexer's identifier tokens, which buys three things
the plan asks for:

* word-boundary matching for free -- ``sleepTimeout`` is one token and is never
  a match for ``sleep``;
* string literals and comments untouched -- they are not identifier tokens;
* an exactly preserved Java token count, because every substitution is
  one identifier token for one identifier token.

Intervention A (cue *removal*) is feature-space only and lives in ``cv.py``.
"""
from __future__ import annotations

from .javalex import tokenize


def rename_code(src: str, mapping: dict[str, str]) -> str:
    """Apply an identifier->identifier renaming to Java source."""
    if not mapping:
        return src
    out: list[str] = []
    for t in tokenize(src):
        if t.kind == "ident":
            out.append(mapping.get(t.text, t.text))
        else:
            out.append(t.text)
    return "".join(out)


def invert_map(mapping: dict[str, str]) -> dict[str, str]:
    """The renaming is a bijection, so it inverts.  Used by the audit."""
    inv: dict[str, str] = {}
    for k, v in mapping.items():
        if v in inv:
            raise ValueError(f"rename map is not injective: {v} <- {inv[v]}, {k}")
        inv[v] = k
    return inv


def rename_stats(src: str, renamed: str) -> dict:
    """Per-row evidence that the rewrite was shape-preserving."""
    from .javalex import code_tokens

    a = code_tokens(src)
    b = code_tokens(renamed)
    return {
        "java_tokens_before": len(a),
        "java_tokens_after": len(b),
        "java_tokens_equal": len(a) == len(b),
        "kinds_equal": [t.kind for t in a] == [t.kind for t in b],
        "n_replaced": sum(1 for x, y in zip(a, b) if x.text != y.text),
    }
