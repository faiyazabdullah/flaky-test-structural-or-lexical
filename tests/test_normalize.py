"""Unit tests for the formatting normalisation (01).

The whitespace artefact this removes gives AUROC 1.000 on the raw release, so a
regression here silently turns every downstream number into a measurement of
which extraction pipeline produced the row.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flaky.normalize import WHITESPACE_FEATURES, normalize_code, whitespace_features


def test_crlf_and_cr_become_lf():
    assert normalize_code("a();\r\nb();\rc();") == "a();\nb();\nc();"


def test_tabs_expand_to_four_spaces():
    assert normalize_code("void f() {\n\tx();\n}") == "void f() {\n    x();\n}"


def test_trailing_whitespace_stripped_per_line():
    assert normalize_code("a();   \nb();\t\n") == "a();\nb();"


def test_leading_and_trailing_blank_lines_stripped():
    assert normalize_code("\n\n  a();\n\n\n") == "a();"


def test_common_indent_removed():
    assert normalize_code("    a();\n        b();\n    c();") == "a();\n    b();\nc();"


def test_column_zero_comment_does_not_defeat_dedent():
    src = "    @Test\n    public void f() {\n//      note\n        x();\n    }"
    out = normalize_code(src)
    assert out.startswith("@Test"), out
    assert "\n//      note\n" in out, out
    assert out.splitlines()[-1] == "}"


def test_output_never_starts_with_whitespace_or_ends_with_newline():
    for src in ["    a();\n", "\t a();\n\n", "//x\n    a();\n", "   \n   a();  \n"]:
        out = normalize_code(src)
        assert not out[:1].isspace(), repr(out)
        assert not out.endswith("\n"), repr(out)


def test_empty_input():
    assert normalize_code("") == ""
    assert normalize_code("\n\n  \n") == ""


def test_whitespace_features_flip_after_normalisation():
    raw = "    @Test\r\n\tvoid f() {}\r\n"
    before = whitespace_features(raw)
    after = whitespace_features(normalize_code(raw))
    assert all(before[k] for k in WHITESPACE_FEATURES)
    assert not any(after[k] for k in WHITESPACE_FEATURES)


def test_relative_indentation_is_preserved():
    src = "    void f() {\n        if (x) {\n            y();\n        }\n    }"
    out = normalize_code(src)
    assert out == "void f() {\n    if (x) {\n        y();\n    }\n}"
