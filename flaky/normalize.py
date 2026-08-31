"""Formatting normalisation for stage 01.

Not a preprocessing nicety.  In the released FlakeBench CSV the two classes are
separable by formatting alone -- no flaky test begins with indentation or ends
with a newline, while 98.9% / 100% of non-flaky tests do.  A subword tokenizer
encodes leading whitespace, so without this step every downstream result is
measuring which extraction pipeline produced the row.
"""
from __future__ import annotations

TAB_WIDTH = 4


def normalize_code(src: str) -> str:
    """CRLF/CR -> LF, tabs -> spaces, strip trailing ws, dedent, strip blank ends."""
    s = src.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_expand_tabs(ln).rstrip() for ln in s.split("\n")]

    # Drop leading/trailing blank lines before measuring the common indent, so a
    # blank first line cannot make the common indent zero.
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return ""

    indent = _common_indent(lines)
    if indent:
        lines = [ln[indent:] if ln.strip() else ln for ln in lines]

    # A single line at column 0 -- typically a commented-out line the author
    # never re-indented -- makes the common indent 0 and leaves the whole method
    # shifted right.  42 rows in the release look like this, all of them
    # non-flaky, so the artefact would survive in weakened form.  Shift left by
    # whatever the first line still carries, clamping each line at column 0.
    first = len(lines[0]) - len(lines[0].lstrip(" "))
    if first:
        lines = [ln[min(first, len(ln) - len(ln.lstrip(" "))):] if ln.strip() else ln
                 for ln in lines]

    return "\n".join(lines)


def _expand_tabs(line: str) -> str:
    """Tab expansion to the next multiple of TAB_WIDTH (str.expandtabs semantics)."""
    return line.expandtabs(TAB_WIDTH)


def _common_indent(lines: list[str]) -> int:
    widths = [len(ln) - len(ln.lstrip(" ")) for ln in lines if ln.strip()]
    return min(widths) if widths else 0


# --- the artefact features, kept as a named set so 01 can assert on them ----

WHITESPACE_FEATURES = (
    "ws_leading_indent",
    "ws_ends_with_newline",
    "ws_contains_tab",
    "ws_contains_crlf",
)


def whitespace_features(src: str) -> dict[str, int]:
    """The four booleans that give AUROC 1.000 on the raw CSV."""
    return {
        "ws_leading_indent": int(bool(src) and src[0] in " \t"),
        "ws_ends_with_newline": int(src.endswith("\n")),
        "ws_contains_tab": int("\t" in src),
        "ws_contains_crlf": int("\r" in src),
    }
