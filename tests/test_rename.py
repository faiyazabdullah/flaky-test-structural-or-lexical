"""Unit tests for the renaming pass.  Required by 03; a silent bug here
invalidates every transfer number downstream."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flaky.cues import build_rename_map, build_static_vocab
from flaky.javalex import code_tokens, identifiers
from flaky.rename import invert_map, rename_code, rename_stats

MAP = {"Thread": "Util", "sleep": "pause", "CountDownLatch": "Gate", "HashMap": "MapA",
       "currentTimeMillis": "readCounter"}


def test_cue_inside_string_literal_is_untouched():
    src = 'String s = "Thread.sleep(10)"; Thread.sleep(10);'
    got = rename_code(src, MAP)
    assert '"Thread.sleep(10)"' in got, got
    assert "Util.pause(10);" in got, got


def test_cue_inside_comment_is_untouched():
    src = "// call Thread.sleep here\n/* HashMap note */\nThread.sleep(1);"
    got = rename_code(src, MAP)
    assert "// call Thread.sleep here" in got
    assert "/* HashMap note */" in got
    assert "Util.pause(1);" in got


def test_substring_of_longer_identifier_is_untouched():
    src = "int sleepTimeout = 5; mySleep(); sleeper.sleep();"
    got = rename_code(src, MAP)
    assert "sleepTimeout" in got
    assert "mySleep()" in got
    assert "sleeper.pause()" in got, got


def test_same_cue_twice_maps_consistently():
    src = "Thread.sleep(1); Thread.sleep(2);"
    got = rename_code(src, MAP)
    assert got == "Util.pause(1); Util.pause(2);", got


def test_java_token_count_unchanged():
    src = ('@Test public void t() throws Exception {\n'
           '  CountDownLatch l = new CountDownLatch(1);\n'
           '  HashMap<String,Integer> m = new HashMap<>();\n'
           '  long t0 = System.currentTimeMillis();\n'
           '  Thread.sleep(10);\n'
           '  assertEquals(1, m.size());\n}')
    st = rename_stats(src, rename_code(src, MAP))
    assert st["java_tokens_equal"], st
    assert st["kinds_equal"], st
    assert st["n_replaced"] == 7, st  # CountDownLatch x2, HashMap x2, currentTimeMillis, Thread, sleep


def test_text_block_is_untouched():
    src = 'String s = """\n  Thread.sleep\n  """; Thread.sleep(1);'
    got = rename_code(src, MAP)
    assert "  Thread.sleep\n" in got
    assert "Util.pause(1);" in got


def test_char_literal_with_quote():
    src = "char q = '\"'; Thread.sleep(1);"
    got = rename_code(src, MAP)
    assert got == "char q = '\"'; Util.pause(1);", got


def test_map_is_a_bijection_and_inverts():
    inv = invert_map(MAP)
    assert inv["Util"] == "Thread"
    src = "Thread.sleep(1);"
    assert rename_code(rename_code(src, MAP), inv) == src


def test_built_map_never_collides_with_corpus_identifiers():
    corpus = {"Thread", "sleep", "Util", "pause", "Gate", "MapA", "foo", "bar"}
    vocab = build_static_vocab(corpus)
    cues = sorted({c for v in vocab.values() for c in v})
    m = build_rename_map(cues, corpus)
    assert len(set(m.values())) == len(m)
    assert not (set(m.values()) & corpus), set(m.values()) & corpus


def test_keywords_are_never_renamed():
    corpus = {"synchronized", "volatile", "foo"}
    m = build_rename_map(["synchronized", "volatile", "foo"], corpus)
    assert "synchronized" not in m and "volatile" not in m
    src = "synchronized (o) { volatile int x; }"
    assert rename_code(src, m) == src


def test_renaming_preserves_identifier_arity_and_structure():
    src = "assertEquals(expected, map.keySet().size());"
    m = {"keySet": "firstView"}
    got = rename_code(src, m)
    assert got == "assertEquals(expected, map.firstView().size());"
    assert len(code_tokens(src)) == len(code_tokens(got))
