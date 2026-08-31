"""A small, dependency-free Java lexer.

Used by the renaming pass (03), the token-distance metric (05) and the cue
miner (03).  Renaming *must not* regex over raw source -- a cue inside a string
literal or a comment has to survive untouched, and ``sleepTimeout`` must not
become ``pauseTimeout``.  Lexing is how that guarantee is made rather than
hoped for.

The lexer is deliberately permissive: FlakeBench rows are method fragments and
some are transpiled Xtend, so it must never raise on odd input.  Anything it
cannot classify becomes an ``other`` token, which the renamer leaves alone.
"""
from __future__ import annotations

from dataclasses import dataclass

IDENT_START = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_$")
IDENT_CONT = IDENT_START | set("0123456789")
DIGITS = set("0123456789")

JAVA_KEYWORDS = frozenset(
    """abstract assert boolean break byte case catch char class const continue default do
    double else enum extends final finally float for goto if implements import instanceof
    int interface long native new package private protected public return short static
    strictfp super switch synchronized this throw throws transient try void volatile while
    true false null var record yield sealed permits non-sealed""".split()
)


@dataclass(frozen=True)
class Token:
    kind: str  # ident | keyword | number | string | char | comment | ws | op | other
    text: str
    start: int
    end: int


def tokenize(src: str) -> list[Token]:
    """Lex ``src`` into tokens whose ``text`` concatenates back to ``src`` exactly."""
    toks: list[Token] = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]

        # whitespace
        if c in " \t\n\r\f\v":
            j = i + 1
            while j < n and src[j] in " \t\n\r\f\v":
                j += 1
            toks.append(Token("ws", src[i:j], i, j))
            i = j
            continue

        # comments
        if c == "/" and i + 1 < n:
            if src[i + 1] == "/":
                j = src.find("\n", i)
                j = n if j < 0 else j
                toks.append(Token("comment", src[i:j], i, j))
                i = j
                continue
            if src[i + 1] == "*":
                j = src.find("*/", i + 2)
                j = n if j < 0 else j + 2
                toks.append(Token("comment", src[i:j], i, j))
                i = j
                continue

        # text block """ ... """
        if c == '"' and src.startswith('"""', i):
            j = i + 3
            while j < n:
                if src[j] == "\\":
                    j += 2
                    continue
                if src.startswith('"""', j):
                    j += 3
                    break
                j += 1
            else:
                j = n
            toks.append(Token("string", src[i:j], i, j))
            i = j
            continue

        # string / char literal
        if c in "\"'":
            quote = c
            j = i + 1
            while j < n:
                if src[j] == "\\":
                    j += 2
                    continue
                if src[j] == quote:
                    j += 1
                    break
                if src[j] == "\n":  # unterminated -- stop at the line end
                    break
                j += 1
            else:
                j = n
            toks.append(Token("string" if quote == '"' else "char", src[i:j], i, j))
            i = j
            continue

        # number
        if c in DIGITS or (c == "." and i + 1 < n and src[i + 1] in DIGITS):
            j = i
            while j < n and (src[j] in IDENT_CONT or src[j] == "."):
                # exponent sign
                if src[j] in "eE" and j + 1 < n and src[j + 1] in "+-" and src[i] != "0":
                    j += 2
                    continue
                j += 1
            toks.append(Token("number", src[i:j], i, j))
            i = j
            continue

        # identifier / keyword
        if c in IDENT_START:
            j = i + 1
            while j < n and src[j] in IDENT_CONT:
                j += 1
            text = src[i:j]
            toks.append(Token("keyword" if text in JAVA_KEYWORDS else "ident", text, i, j))
            i = j
            continue

        # anything else is a single-character operator/punctuation token
        kind = "op" if c in "+-*/%=<>!&|^~?:;,.(){}[]@" else "other"
        toks.append(Token(kind, c, i, i + 1))
        i += 1

    return toks


def code_tokens(src: str) -> list[Token]:
    """Tokens that carry program content -- whitespace and comments dropped."""
    return [t for t in tokenize(src) if t.kind not in ("ws", "comment")]


def identifiers(src: str) -> list[str]:
    """Identifier texts in source order (keywords excluded)."""
    return [t.text for t in tokenize(src) if t.kind == "ident"]


def java_token_count(src: str) -> int:
    return len(code_tokens(src))


def token_texts(src: str) -> list[str]:
    return [t.text for t in code_tokens(src)]


def edit_distance(a: list[str], b: list[str]) -> int:
    """Levenshtein over token sequences (used for the minimal-pair diff size)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ta in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, tb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ta != tb))
        prev = cur
    return prev[-1]
