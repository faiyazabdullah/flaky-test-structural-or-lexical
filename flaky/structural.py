"""04 -- structural property labels, computed without the flakiness label.

An intra-procedural, taint-style reaching-definitions pass over a tree-sitter
Java parse.  Three properties, exactly as specified in ``plan/04``:

``P_ASYNC``  an async dispatch reaches an assertion with no intervening
             synchronisation *on the path*.  This is control-flow reachability
             with a kill set, not data flow: the common real pattern
             (``ex.submit(task); assertTrue(counter.get() > 0)``) carries no
             def-use chain from the dispatch to the assertion, and a pure
             data-flow reading would miss precisely the cases that matter.
``P_UNORD``  a value originating from unordered iteration reaches an assertion
             *argument* -- data flow, as the property is worded.
``P_CLOCK``  a clock read reaches an assertion argument -- data flow.

Branches merge existentially (union of taint, OR of liveness), because the
property asks whether the value reaches an assertion on *some* path.  Loops are
iterated to a fixpoint over the (small) lattice.

Aliasing, field access and lambda capture are approximated.  That is acceptable
only because ``scripts/s04_structural.py`` measures the approximation against a
hand audit rather than assuming it.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass, field, replace
from typing import Iterable

# --------------------------------------------------------------------------
# vocabularies -- exactly the plan's lists
# --------------------------------------------------------------------------

ASYNC_DISPATCH = frozenset({"submit", "execute", "start", "thenApply", "runAsync",
                            "supplyAsync", "schedule"})

# ``await``, ``sleep`` and ``invokeAll`` are unambiguous barriers.  ``get`` and
# ``join`` are not: ``future.get()`` is a barrier but ``counter.get()`` and
# ``map.get(k)`` are not.  They therefore count as synchronisation only at zero
# arity *and* on a receiver resolvable as future-like.  Treating every
# ``get()`` as a barrier would silently gut recall on P_ASYNC, since
# ``pool.submit(t); assertTrue(counter.get() > 0)`` is the commonest real shape
# of the property.  The cost is a receiver we cannot resolve (a field, say),
# where the barrier is missed -- an approximation the hand audit measures.
SYNC_ANY_ARITY = frozenset({"await", "sleep", "invokeAll", "awaitTermination"})
# Arities at which the name is the blocking Java API rather than something else:
# ``Future.get()`` / ``Future.get(timeout, unit)``; ``Thread.join()`` /
# ``Thread.join(millis)``.  ``map.get(k)`` and ``String.join(sep, parts)`` are
# excluded by arity, the rest by the future-like receiver check.
SYNC_ARITIES = {"get": (0, 2), "join": (0, 1)}
SYNC_ZERO_ARITY = frozenset(SYNC_ARITIES)
FUTURE_LIKE_TYPES = frozenset({"Future", "CompletableFuture", "CompletionStage",
                               "ScheduledFuture", "FutureTask", "ForkJoinTask",
                               "ListenableFuture", "Thread", "Task"})

UNORD_SOURCE_CALLS = frozenset({"keySet", "entrySet", "values", "listFiles", "toArray"})
RECEIVER_GATED_UNORD_CALLS = frozenset({"values", "toArray"})
UNORDERED_TYPES = frozenset({"HashMap", "HashSet", "Hashtable", "HashTable",
                             "ConcurrentHashMap", "IdentityHashMap", "WeakHashMap",
                             "HashMultimap", "HashBiMap", "ConcurrentHashSet",
                             "CopyOnWriteArraySet"})
ORDERED_TYPES = frozenset({"LinkedHashMap", "LinkedHashSet", "TreeMap", "TreeSet",
                           "SortedMap", "SortedSet", "NavigableMap", "NavigableSet",
                           "ImmutableSortedMap", "ImmutableSortedSet", "ArrayList",
                           "LinkedList", "List", "ImmutableList", "ArrayDeque"})
# ``Map`` and ``Set`` promise no iteration order in their interface contract, so
# a variable of that declared type is unordered unless its initialiser names a
# type that *does* promise one (``new LinkedHashMap<>()``).  ``Collection`` and
# ``Iterable`` are too broad to treat this way and are deliberately excluded.
AMBIGUOUS_UNORDERED_TYPES = frozenset({"Map", "Set"})

# Static factories that hand back an unordered collection:
# ``HashMultimap.create()``, ``Sets.newHashSet()``, ``Maps.newConcurrentMap()``.
UNORD_FACTORY_NAMES = frozenset({"newHashSet", "newHashMap", "newConcurrentMap",
                                 "newIdentityHashMap", "newSetFromMap", "newKeySet",
                                 "newConcurrentHashSet"})

CLOCK_CALLS = frozenset({"currentTimeMillis", "nanoTime", "now"})
CLOCK_TYPES = frozenset({"Date", "GregorianCalendar"})

ITER_CALLS = frozenset({"iterator", "stream", "forEach", "parallelStream", "spliterator"})

# Operations on an unordered collection whose result does not depend on
# iteration order.  ``assertEquals(2, map.size())`` is not order-sensitive;
# ``assertEquals(s, map.toString())`` is.  Set/Map ``equals`` is order-
# independent too, which is why asserting on the collection itself does not
# count -- only a value that escaped through an order-exposing operation does.
ORDER_INSENSITIVE_CALLS = frozenset({"size", "isEmpty", "contains", "containsKey",
                                     "containsValue", "containsAll", "get", "getOrDefault",
                                     "equals", "hashCode", "length", "count", "put",
                                     "putAll", "add", "addAll", "remove", "clear"})

# Calls that render a whole collection, so the result *does* depend on its
# iteration order.  Anything not listed here and not an iteration is treated as
# opaque: passing a HashMap to a service method and asserting on the response
# is not evidence that the assertion depends on map order, and treating it as
# such was the single source of every P_UNORD false positive in the hand audit.
ORDER_EXPOSING_CALLS = frozenset({"toString", "toJSONString", "writeValueAsString", "toJson",
                                  "toPrettyString", "asText", "serialize", "format",
                                  "toStringHelper", "mkString", "toArray", "toList",
                                  "collect", "reduce", "flatMap", "map", "sorted"})

ASSERT_PREFIXES = ("assert", "expect")
ASSERT_EXACT = frozenset({"verify", "fail", "checkThat", "shouldBe", "shouldEqual"})

# Calls that return a mock proxy.  A method invoked on one records or verifies
# an interaction rather than performing it, so ``verify(mock).start()`` is not
# an async dispatch and ``when(mock.get())`` is not a barrier.
MOCK_WRAPPERS = frozenset({"verify", "when", "doReturn", "doThrow", "doAnswer", "doNothing",
                           "given", "stub", "inOrder", "spy", "mock"})

PROPERTIES = ("P_ASYNC", "P_UNORD", "P_CLOCK")


def is_assertion(name: str) -> bool:
    """Method names only.  ``Assert`` is a class and must not match -- the
    analyser passes the invocation's ``name`` field, so ``Assert.assertTrue``
    arrives here as ``assertTrue``."""
    return name.startswith(ASSERT_PREFIXES) or name in ASSERT_EXACT


def is_sync(name: str, n_args: int, receiver_is_future: bool = True) -> bool:
    """``receiver_is_future`` defaults True so the predicate can be reasoned
    about on its own; the analyser always passes the resolved value."""
    if name in SYNC_ANY_ARITY:
        return True
    if name not in SYNC_ARITIES:
        return False
    return n_args in SYNC_ARITIES[name] and receiver_is_future


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------

WRAPPER_PREFIX = "class Synth {\n"
WRAPPER_SUFFIX = "\n}\n"


@functools.lru_cache(maxsize=1)
def get_parser():
    """tree-sitter-java parser.  Cached; constructing it is not free."""
    import tree_sitter_java
    from tree_sitter import Language, Parser

    lang = Language(tree_sitter_java.language())
    try:
        return Parser(lang)
    except TypeError:  # older tree-sitter API
        p = Parser()
        p.set_language(lang)
        return p


def wrap(code: str) -> str:
    """Test methods are fragments; give them a class to live in before parsing."""
    return WRAPPER_PREFIX + code + WRAPPER_SUFFIX


def _repair_candidates(code: str) -> list[tuple[str, str]]:
    """Suffixes that close a method the dataset cut short.

    6.5% of FlakeBench rows are truncated mid-body -- the extraction stopped at
    a length limit -- and they are 99% non-flaky, so dropping them as parse
    failures would bias the labels in exactly the direction the plan warns
    about.  Closing the open brackets recovers the statements that *are*
    present without inventing any content.  Rows that need a repair are flagged
    so the rate is reported rather than hidden.
    """
    from .javalex import code_tokens

    toks = code_tokens(code)
    if not toks:
        return []
    b = sum(1 for t in toks if t.text == "{") - sum(1 for t in toks if t.text == "}")
    p = sum(1 for t in toks if t.text == "(") - sum(1 for t in toks if t.text == ")")
    b, p = max(b, 0), max(p, 0)
    last = toks[-1].text
    cands: list[tuple[str, str]] = []
    if p or b:
        cands.append(("close_brackets", ")" * p + "}" * b))
        cands.append(("close_brackets_semicolon", ")" * p + ";" + "}" * b))
    if last not in (";", "}", "{"):
        cands.append(("semicolon", ";"))
        cands.append(("empty_body", ")" * p + "{}"))
    return cands


def _truncate_to_last_statement(code: str) -> str | None:
    """Cut back to the last statement that finished, then close the braces.

    Truncation usually lands in the middle of a statement (``aclEntry(A, B),``),
    where appending brackets cannot help.  Rewinding to the last ``;`` or ``}``
    at bracket depth zero keeps every statement that is intact and discards the
    partial one, which is the most that can honestly be recovered.
    """
    from .javalex import tokenize

    depth_paren = depth_brack = depth_brace = 0
    cut_end = None
    cut_brace = 0
    for t in tokenize(code):
        if t.kind in ("ws", "comment", "string", "char"):
            continue
        c = t.text
        if c == "(":
            depth_paren += 1
        elif c == ")":
            depth_paren -= 1
        elif c == "[":
            depth_brack += 1
        elif c == "]":
            depth_brack -= 1
        elif c == "{":
            depth_brace += 1
        elif c == "}":
            depth_brace -= 1
        if (c in (";", "}") and depth_paren <= 0 and depth_brack <= 0 and depth_brace >= 1):
            cut_end, cut_brace = t.end, depth_brace
    if cut_end is None or cut_brace <= 0:
        return None
    return code[:cut_end] + "\n" + "}" * cut_brace


def _rewind_and_close(code: str, max_tries: int = 16) -> list[str]:
    """Candidates formed by rewinding to a plausible expression end and closing.

    Handles the case the statement-level rewind cannot: a body truncated inside
    a multi-line argument list (``aclEntry(ACCESS, GROUP, NONE),``), where no
    statement ever completed.  Rewinding to the last complete argument and
    closing the open brackets keeps the call's structure.
    """
    from .javalex import tokenize

    toks = [t for t in tokenize(code) if t.kind not in ("ws", "comment")]
    ends = {")", ";", "}", "]"}
    out: list[str] = []
    for i in range(len(toks) - 1, -1, -1):
        t = toks[i]
        if not (t.text in ends or t.kind in ("ident", "number", "string", "char", "keyword")):
            continue
        prefix = toks[: i + 1]
        b = sum(1 for x in prefix if x.text == "{") - sum(1 for x in prefix if x.text == "}")
        p_ = sum(1 for x in prefix if x.text == "(") - sum(1 for x in prefix if x.text == ")")
        br = sum(1 for x in prefix if x.text == "[") - sum(1 for x in prefix if x.text == "]")
        if b <= 0 or p_ < 0 or br < 0:
            continue
        out.append(code[: t.end] + "]" * br + ")" * p_ + ";" + "}" * b)
        if len(out) >= max_tries:
            break
    return out


def parse(code: str, repair: bool = True):
    """Return ``(tree, source_bytes, ok, repair)``.

    ``ok`` is False if the parse has ERROR or MISSING nodes anywhere.  ``repair``
    names the suffix that made a truncated fragment parse, or ``""``.
    """
    src = wrap(code).encode("utf-8")
    tree = get_parser().parse(src)
    if not _has_error(tree.root_node):
        return tree, src, True, ""
    if repair:
        for name, suffix in _repair_candidates(code):
            src2 = wrap(code + suffix).encode("utf-8")
            tree2 = get_parser().parse(src2)
            if not _has_error(tree2.root_node):
                return tree2, src2, True, name
        cut = _truncate_to_last_statement(code)
        if cut is not None:
            src2 = wrap(cut).encode("utf-8")
            tree2 = get_parser().parse(src2)
            if not _has_error(tree2.root_node):
                return tree2, src2, True, "truncate_to_last_statement"
        for cand in _rewind_and_close(code):
            src2 = wrap(cand).encode("utf-8")
            tree2 = get_parser().parse(src2)
            if not _has_error(tree2.root_node):
                return tree2, src2, True, "rewind_and_close"
    return tree, src, False, ""


def _has_error(node) -> bool:
    if not node.has_error:
        return False
    stack = [node]
    while stack:
        n = stack.pop()
        if n.type == "ERROR" or n.is_missing:
            return True
        if n.has_error:
            stack.extend(n.children)
    return False


# --------------------------------------------------------------------------
# analysis state
# --------------------------------------------------------------------------


@dataclass
class State:
    """May-analysis state at a program point."""

    async_live: bool = False
    unord: frozenset = frozenset()      # local vars holding unordered-derived values
    clock: frozenset = frozenset()      # local vars holding clock-derived values
    containers: frozenset = frozenset()  # local vars holding unordered collections
    futures: frozenset = frozenset()    # local vars holding a Future-like handle

    def merge(self, other: "State") -> "State":
        return State(
            async_live=self.async_live or other.async_live,
            unord=self.unord | other.unord,
            clock=self.clock | other.clock,
            containers=self.containers | other.containers,
            futures=self.futures | other.futures,
        )

    def key(self):
        return (self.async_live, self.unord, self.clock, self.containers, self.futures)


@dataclass
class Hits:
    P_ASYNC: bool = False
    P_UNORD: bool = False
    P_CLOCK: bool = False
    # evidence, kept for the hand audit
    witness: dict = field(default_factory=dict)

    def note(self, prop: str, text: str) -> None:
        setattr(self, prop, True)
        self.witness.setdefault(prop, text[:200])


@dataclass
class Taint:
    """Whether an evaluated expression carries each kind of tainted value."""

    unord: bool = False
    clock: bool = False
    container: bool = False
    future: bool = False

    def __or__(self, other: "Taint") -> "Taint":
        return Taint(self.unord or other.unord, self.clock or other.clock,
                     self.container or other.container, self.future or other.future)


EMPTY = Taint()


# --------------------------------------------------------------------------
# the analyser
# --------------------------------------------------------------------------


class MethodAnalysis:
    def __init__(self, src: bytes, max_loop_iters: int = 4, sleep_is_barrier: bool = True):
        self.src = src
        self.hits = Hits()
        self.max_loop_iters = max_loop_iters
        self.sleep_is_barrier = sleep_is_barrier
        self._budget = 200_000  # guards against pathological nesting

    # -- helpers ----------------------------------------------------------
    def text(self, node) -> str:
        return self.src[node.start_byte:node.end_byte].decode("utf-8", "replace")

    def _spend(self) -> None:
        self._budget -= 1
        if self._budget <= 0:
            raise TimeoutError("analysis budget exhausted")

    # -- entry ------------------------------------------------------------
    def run(self, method_node) -> Hits:
        body = method_node.child_by_field_name("body")
        if body is None:
            return self.hits
        try:
            self.block(body, State())
        except TimeoutError:
            pass
        return self.hits

    # -- statements -------------------------------------------------------
    def block(self, node, st: State) -> State:
        for child in node.named_children:
            st = self.stmt(child, st)
        return st

    def stmt(self, node, st: State) -> State:
        self._spend()
        t = node.type

        if t in ("block", "constructor_body"):
            return self.block(node, st)

        if t == "local_variable_declaration":
            return self.local_decl(node, st)

        if t == "expression_statement":
            child = node.named_children[0] if node.named_children else None
            if child is not None:
                _, st = self.expr(child, st)
            return st

        if t == "if_statement":
            cond = node.child_by_field_name("condition")
            if cond is not None:
                _, st = self.expr(cond, st)
            then = node.child_by_field_name("consequence")
            alt = node.child_by_field_name("alternative")
            s_then = self.stmt(then, st) if then is not None else st
            s_else = self.stmt(alt, st) if alt is not None else st
            return s_then.merge(s_else)

        if t in ("while_statement", "do_statement", "for_statement",
                 "enhanced_for_statement"):
            return self.loop(node, st)

        if t in ("switch_expression", "switch_statement"):
            cond = node.child_by_field_name("condition")
            if cond is not None:
                _, st = self.expr(cond, st)
            body = node.child_by_field_name("body")
            out = st
            if body is not None:
                for grp in body.named_children:
                    out = out.merge(self.stmt(grp, st))
            return out

        if t in ("switch_block_statement_group", "switch_rule"):
            return self.block(node, st)

        if t == "try_statement" or t == "try_with_resources_statement":
            res = node.child_by_field_name("resources")
            if res is not None:
                for r in res.named_children:
                    st = self.stmt(r, st)
            body = node.child_by_field_name("body")
            after = self.stmt(body, st) if body is not None else st
            merged = after
            for child in node.named_children:
                if child.type == "catch_clause":
                    merged = merged.merge(self.stmt(child, st))
                elif child.type == "finally_clause":
                    merged = self.stmt(child, merged)
            return merged

        if t in ("catch_clause", "finally_clause", "synchronized_statement",
                 "labeled_statement"):
            body = node.child_by_field_name("body")
            if body is None:
                body = next((c for c in node.named_children if c.type == "block"), None)
            return self.stmt(body, st) if body is not None else st

        if t == "resource":
            # A try-with-resources resource carries type/name/value directly --
            # it has no variable_declarator child -- so local_decl cannot read
            # it and the resource's initialiser would go unanalysed.
            name_node = node.child_by_field_name("name")
            val = node.child_by_field_name("value")
            type_node = node.child_by_field_name("type")
            taint = EMPTY
            if val is not None:
                taint, st = self.expr(val, st)
            if name_node is not None:
                st = self.bind(st, self.text(name_node), taint,
                               declared_type=self.base_type_name(type_node)
                               if type_node is not None else "",
                               ordered_initialiser=self.is_ordered_new(val))
            elif val is None:
                for c in node.named_children:
                    _, st = self.expr(c, st)
            return st

        if t in ("return_statement", "throw_statement", "yield_statement",
                 "assert_statement"):
            for c in node.named_children:
                _, st = self.expr(c, st)
            return st

        if t in ("local_class_declaration", "class_declaration", "record_declaration",
                 "interface_declaration", "enum_declaration"):
            return st  # nested type bodies are out of scope for this pass

        # anything else: walk children as expressions
        for c in node.named_children:
            if c.is_named:
                _, st = self.expr(c, st)
        return st

    def local_decl(self, node, st: State) -> State:
        decl_type = node.child_by_field_name("type")
        type_name = self.base_type_name(decl_type) if decl_type is not None else ""
        for d in node.named_children:
            if d.type != "variable_declarator":
                continue
            name_node = d.child_by_field_name("name")
            val = d.child_by_field_name("value")
            taint = EMPTY
            if val is not None:
                taint, st = self.expr(val, st)
            if name_node is None:
                continue
            name = self.text(name_node)
            st = self.bind(st, name, taint, declared_type=type_name,
                           initialised=val is not None,
                           ordered_initialiser=self.is_ordered_new(val))
        return st

    def is_ordered_new(self, val) -> bool:
        """``new LinkedHashMap<>()`` -- an initialiser that does promise an order."""
        if val is None or val.type != "object_creation_expression":
            return False
        t = val.child_by_field_name("type")
        return t is not None and self.base_type_name(t) in ORDERED_TYPES

    def bind(self, st: State, name: str, taint: Taint, declared_type: str = "",
             initialised: bool = True, ordered_initialiser: bool = False) -> State:
        """Strong update: an assignment replaces whatever the name held."""
        unord = set(st.unord)
        clock = set(st.clock)
        cont = set(st.containers)
        fut = set(st.futures)
        for s_ in (unord, clock, cont, fut):
            s_.discard(name)
        if taint.unord:
            unord.add(name)
        if taint.clock:
            clock.add(name)
        if taint.container:
            cont.add(name)
        if taint.future:
            fut.add(name)
        if declared_type:
            if declared_type in UNORDERED_TYPES:
                cont.add(name)
            elif declared_type in AMBIGUOUS_UNORDERED_TYPES and not ordered_initialiser:
                # The Map/Set contract promises no iteration order, so an
                # opaquely-initialised one counts; ``new LinkedHashMap<>()``
                # does promise one and does not.
                cont.add(name)
            if declared_type in FUTURE_LIKE_TYPES:
                fut.add(name)
        return State(st.async_live, frozenset(unord), frozenset(clock), frozenset(cont),
                     frozenset(fut))

    def loop(self, node, st: State) -> State:
        """Iterate the body to a fixpoint; the zero-trip path is included by
        merging the entry state."""
        out = st
        for _ in range(self.max_loop_iters):
            cur = out
            if node.type == "enhanced_for_statement":
                cur = self.enhanced_for_head(node, cur)
            else:
                for fname in ("init", "condition", "update"):
                    n = node.child_by_field_name(fname)
                    if n is not None:
                        if n.type == "local_variable_declaration":
                            cur = self.local_decl(n, cur)
                        else:
                            _, cur = self.expr(n, cur)
            body = node.child_by_field_name("body")
            if body is not None:
                cur = self.stmt(body, cur)
            merged = out.merge(cur)
            if merged.key() == out.key():
                return merged
            out = merged
        return out

    def enhanced_for_head(self, node, st: State) -> State:
        """``for (T x : coll)`` -- x inherits the taint of ``coll``."""
        coll = node.child_by_field_name("value")
        taint = EMPTY
        if coll is not None:
            taint, st = self.expr(coll, st)
            if taint.container:
                taint = taint | Taint(unord=True)
        name_node = node.child_by_field_name("name")
        type_node = node.child_by_field_name("type")
        tname = self.base_type_name(type_node) if type_node is not None else ""
        if name_node is not None:
            st = self.bind(st, self.text(name_node), taint, declared_type=tname)
        return st

    # -- expressions ------------------------------------------------------
    def expr(self, node, st: State) -> tuple[Taint, State]:
        """Post-order evaluation.  Returns the taint of the value and the state
        after evaluating side effects.  Post-order matters: the arguments of a
        call are evaluated -- and can therefore kill async liveness -- before
        the call itself is seen."""
        self._spend()
        if node is None:
            return EMPTY, st
        t = node.type

        if t == "identifier":
            name = self.text(node)
            return Taint(name in st.unord, name in st.clock, name in st.containers,
                         name in st.futures), st

        if t in ("decimal_integer_literal", "string_literal", "character_literal",
                 "true", "false", "null_literal", "decimal_floating_point_literal",
                 "hex_integer_literal", "line_comment", "block_comment"):
            return EMPTY, st

        if t == "method_invocation":
            return self.method_invocation(node, st)

        if t == "object_creation_expression":
            return self.object_creation(node, st)

        if t == "assignment_expression":
            left = node.child_by_field_name("left")
            right = node.child_by_field_name("right")
            taint, st = self.expr(right, st)
            if left is not None and left.type == "identifier":
                st = self.bind(st, self.text(left), taint)
            elif left is not None:
                _, st = self.expr(left, st)
            return taint, st

        if t == "lambda_expression":
            # Capture is approximated by inlining the body in the current state.
            body = node.child_by_field_name("body")
            if body is None:
                return EMPTY, st
            if body.type == "block":
                st = self.stmt(body, st)
                return EMPTY, st
            return self.expr(body, st)

        if t == "field_access":
            obj = node.child_by_field_name("object")
            taint, st = self.expr(obj, st) if obj is not None else (EMPTY, st)
            fld = node.child_by_field_name("field")
            if fld is not None and self.text(fld) in ("now",):
                taint = taint | Taint(clock=True)
            return taint, st

        if t in ("ternary_expression",):
            cond = node.child_by_field_name("condition")
            _, st = self.expr(cond, st) if cond is not None else (EMPTY, st)
            a = node.child_by_field_name("consequence")
            b = node.child_by_field_name("alternative")
            ta, st_a = self.expr(a, st) if a is not None else (EMPTY, st)
            tb, st_b = self.expr(b, st) if b is not None else (EMPTY, st)
            return ta | tb, st_a.merge(st_b)

        # default: fold over named children, unioning taint
        taint = EMPTY
        for c in node.named_children:
            ct, st = self.expr(c, st)
            taint = taint | ct
        return taint, st

    def method_invocation(self, node, st: State) -> tuple[Taint, State]:
        obj = node.child_by_field_name("object")
        name_node = node.child_by_field_name("name")
        args_node = node.child_by_field_name("arguments")
        name = self.text(name_node) if name_node is not None else ""

        # receiver first, then arguments -- source order
        recv_taint, st = self.expr(obj, st) if obj is not None else (EMPTY, st)
        arg_taints: list[Taint] = []
        args = [a for a in args_node.named_children] if args_node is not None else []
        for a in args:
            at, st = self.expr(a, st)
            arg_taints.append(at)
        combined = functools.reduce(lambda x, y: x | y, arg_taints, recv_taint)

        # --- sinks -------------------------------------------------------
        if is_assertion(name):
            if st.async_live:
                self.hits.note("P_ASYNC", self.text(node))
            for at in arg_taints or [recv_taint]:
                if at.unord:
                    self.hits.note("P_UNORD", self.text(node))
                if at.clock:
                    self.hits.note("P_CLOCK", self.text(node))
            # Asserting on the collection *itself* -- assertEquals(expected,
            # someSet) -- deliberately does not count: Set and Map equality is
            # order-independent. Only a value produced by iterating it does,
            # which is what the taint rules below encode.

        # A call on the result of verify(...)/when(...) is a mock interaction,
        # not a real one: it neither dispatches nor synchronises.
        on_mock = obj is not None and self.is_mock_expression(obj)

        # --- kills before gens: a sync call ends the async window --------
        if on_mock or (name == "sleep" and not self.sleep_is_barrier):
            pass
        elif is_sync(name, len(args), receiver_is_future=recv_taint.future):
            st = replace(st, async_live=False)

        # --- gens --------------------------------------------------------
        if name in ASYNC_DISPATCH and not on_mock:
            st = replace(st, async_live=True)

        # What the call's result carries.  Inherited taint is computed first,
        # then stripped where the call does not actually pass the value
        # through, then the taint the call itself produces is added.  Doing it
        # in that order matters: a factory must not have its own container
        # taint stripped by the opaque-call rule below.
        produced = Taint()
        if name in ASYNC_DISPATCH:
            produced = produced | Taint(future=True)
        if name in CLOCK_CALLS:
            produced = produced | Taint(clock=True)
        if name in UNORD_SOURCE_CALLS:
            if name in RECEIVER_GATED_UNORD_CALLS:
                # ``values`` and ``toArray`` are not exclusively collection
                # methods -- ``testSubscriber.values()`` is a list of emitted
                # items, not a Map view -- so they only count on a receiver we
                # can resolve as an unordered collection.
                if recv_taint.container or recv_taint.unord:
                    produced = produced | Taint(unord=True)
            else:
                produced = produced | Taint(unord=True)
        if name in ITER_CALLS and (recv_taint.container or recv_taint.unord):
            produced = produced | Taint(unord=True)
        if name in UNORD_FACTORY_NAMES or (
                obj is not None and obj.type == "identifier"
                and self.text(obj) in UNORDERED_TYPES):
            produced = produced | Taint(container=True)
        elif combined.container and name in ORDER_EXPOSING_CALLS:
            # A whole collection rendered by an order-exposing call lets its
            # iteration order escape into the result.
            produced = produced | Taint(unord=True)

        inherited = combined
        if name not in UNORD_SOURCE_CALLS and name not in ITER_CALLS:
            # Container-ness does not survive an opaque call: the result of
            # ``service.handle(map)`` is a response, not the map, and need not
            # depend on the map's iteration order. Assuming otherwise produced
            # every P_UNORD false positive in the hand audit.
            inherited = Taint(inherited.unord, inherited.clock, False, inherited.future)
        if name in ORDER_INSENSITIVE_CALLS:
            # The result of map.get(k) or set.size() is a plain value; an
            # ordinary accessor chained onto it must not fire.
            inherited = Taint(False, inherited.clock, False, inherited.future)

        out = inherited | produced
        return out, st

    def object_creation(self, node, st: State) -> tuple[Taint, State]:
        type_node = node.child_by_field_name("type")
        tname = self.base_type_name(type_node) if type_node is not None else ""
        args_node = node.child_by_field_name("arguments")
        taint = EMPTY
        if args_node is not None:
            for a in args_node.named_children:
                at, st = self.expr(a, st)
                taint = taint | at
        if tname in UNORDERED_TYPES:
            taint = taint | Taint(container=True)
        if tname in CLOCK_TYPES:
            taint = taint | Taint(clock=True)
        if tname in FUTURE_LIKE_TYPES:
            taint = taint | Taint(future=True)
        return taint, st

    def is_mock_expression(self, node) -> bool:
        """True if ``node`` evaluates to a Mockito proxy (``verify(x)``,
        ``when(y)``, or a chain rooted at one)."""
        n = node
        for _ in range(4):
            if n is None:
                return False
            if n.type == "method_invocation":
                name_node = n.child_by_field_name("name")
                if name_node is not None and self.text(name_node) in MOCK_WRAPPERS:
                    return True
                n = n.child_by_field_name("object")
                continue
            if n.type in ("field_access", "parenthesized_expression"):
                n = n.child_by_field_name("object") or (
                    n.named_children[0] if n.named_children else None)
                continue
            return False
        return False

    def base_type_name(self, node) -> str:
        """``Map<String, List<X>>`` -> ``Map``; ``java.util.HashMap`` -> ``HashMap``."""
        if node is None:
            return ""
        n = node
        while n.type in ("generic_type", "array_type", "annotated_type"):
            child = n.child_by_field_name("type") or (
                n.named_children[0] if n.named_children else None)
            if child is None or child is n:
                break
            n = child
        txt = self.text(n)
        txt = txt.split("<")[0].strip()
        return txt.split(".")[-1]


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------


def find_methods(root) -> list:
    """Method declarations inside the synthetic wrapper class."""
    out = []
    stack = [root]
    while stack:
        n = stack.pop()
        if n.type in ("method_declaration", "constructor_declaration"):
            out.append(n)
            continue
        stack.extend(n.children)
    return out


def analyse(code: str) -> dict:
    """Label one test method with the three structural properties.

    Returns ``{'parse_ok', 'parse_repair', 'P_ASYNC', 'P_UNORD', 'P_CLOCK',
    'n_methods', 'witness'}``.

    tree-sitter is error-tolerant, so a row that still fails to parse after
    repair is analysed on whatever the parser recovered rather than forced to
    all-False.  Forcing negatives would bias the labels: the unparsed rows are
    almost entirely non-flaky.  ``parse_ok`` is carried into
    ``data/structural.csv`` so downstream code can filter if it wants to.
    """
    try:
        tree, src, ok, repair = parse(code)
    except Exception as exc:  # pragma: no cover - tree-sitter should not raise
        return {"parse_ok": False, "parse_repair": "", "n_methods": 0, "error": repr(exc),
                **{p: False for p in PROPERTIES}, "P_ASYNC_sleep_kills": False, "witness": {}}

    methods = find_methods(tree.root_node)
    hits = Hits()
    sleep_variant = False
    for m in methods:
        h = MethodAnalysis(src).run(m)
        for p in PROPERTIES:
            if getattr(h, p):
                hits.note(p, h.witness.get(p, ""))
        h2 = MethodAnalysis(src, sleep_is_barrier=False).run(m)
        sleep_variant = sleep_variant or h2.P_ASYNC
    return {
        "parse_ok": ok,
        "parse_repair": repair,
        "n_methods": len(methods),
        **{p: bool(getattr(hits, p)) for p in PROPERTIES},
        # Diagnostic, not a property and not part of any stopping rule: the
        # canonical async-wait flaky test is "dispatch; sleep; assert", and the
        # plan's definition puts `sleep` in the kill set, so that shape is
        # excluded from P_ASYNC by construction. Reported so the write-up can
        # say how much of the category the definition rules out.
        "P_ASYNC_sleep_kills": bool(sleep_variant),
        "witness": hits.witness,
    }


def analyse_many(codes: Iterable[str]) -> list[dict]:
    return [analyse(c) for c in codes]
