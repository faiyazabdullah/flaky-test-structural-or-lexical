"""Unit tests for the structural-property analyser.

Required by 00: renaming and structural analysis are where a silent bug
invalidates everything downstream.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flaky.structural import analyse, is_assertion, is_sync


def M(body: str) -> str:
    return "@Test\npublic void t() throws Exception {\n" + body + "\n}"


# -- P_ASYNC -----------------------------------------------------------------

def test_async_dispatch_reaching_assertion():
    r = analyse(M("""
    executor.submit(task);
    assertTrue(counter.get() > 0);
    """))
    assert r["parse_ok"] and r["P_ASYNC"]


def test_async_dispatch_killed_by_join():
    r = analyse(M("""
    Future<?> f = executor.submit(task);
    f.join();
    assertTrue(done);
    """))
    assert r["parse_ok"] and not r["P_ASYNC"]


def test_async_dispatch_killed_by_zero_arg_get_in_the_assertion_argument():
    # arguments are evaluated before the call, so f.get() is an intervening
    # synchronisation even though it sits inside the assertion
    r = analyse(M("""
    Future<Integer> f = executor.submit(task);
    assertEquals(1, f.get());
    """))
    assert not r["P_ASYNC"]


def test_map_get_with_an_argument_is_not_synchronisation():
    r = analyse(M("""
    executor.submit(task);
    assertEquals(1, map.get("k"));
    """))
    assert r["P_ASYNC"]


def test_sleep_is_synchronisation():
    r = analyse(M("""
    executor.execute(task);
    Thread.sleep(100);
    assertTrue(done);
    """))
    assert not r["P_ASYNC"]


def test_no_assertion_means_no_async_property():
    r = analyse(M("executor.submit(task);\n    log.info(\"x\");"))
    assert not r["P_ASYNC"]


def test_assertion_before_dispatch_does_not_count():
    r = analyse(M("""
    assertTrue(ready);
    executor.submit(task);
    """))
    assert not r["P_ASYNC"]


def test_async_inside_a_branch_still_reaches():
    r = analyse(M("""
    if (flag) {
        pool.execute(task);
    }
    assertTrue(counter.get() > 0);
    """))
    assert r["P_ASYNC"]


def test_sync_on_only_one_branch_leaves_a_live_path():
    r = analyse(M("""
    pool.execute(task);
    if (flag) {
        latch.await();
    }
    assertTrue(done);
    """))
    assert r["P_ASYNC"], "the else path has no synchronisation"


def test_sync_on_every_path_kills_it():
    r = analyse(M("""
    pool.execute(task);
    if (flag) { latch.await(); } else { latch.await(); }
    assertTrue(done);
    """))
    assert not r["P_ASYNC"]


# -- P_UNORD -----------------------------------------------------------------

def test_keyset_reaching_an_assertion_argument():
    r = analyse(M("""
    Set<String> ks = map.keySet();
    assertEquals(expected, ks);
    """))
    assert r["P_UNORD"]


def test_hashmap_iteration_taints_the_loop_variable():
    r = analyse(M("""
    HashMap<String, Integer> m = new HashMap<>();
    for (String k : m) {
        assertEquals(1, k);
    }
    """))
    assert r["P_UNORD"]


def test_linked_hashmap_is_ordered_and_does_not_taint():
    r = analyse(M("""
    Map<String, Integer> m = new LinkedHashMap<>();
    for (String k : m) {
        assertEquals(1, k);
    }
    """))
    assert not r["P_UNORD"]


def test_unordered_value_not_reaching_an_assertion():
    r = analyse(M("""
    Set<String> ks = map.keySet();
    log.info(ks.toString());
    assertTrue(flag);
    """))
    assert not r["P_UNORD"]


def test_reassignment_kills_taint():
    r = analyse(M("""
    Object v = map.keySet();
    v = "plain";
    assertEquals("plain", v);
    """))
    assert not r["P_UNORD"]


def test_taint_flows_through_a_call_argument():
    r = analyse(M("""
    List<String> l = new ArrayList<>(map.keySet());
    assertEquals(expected, l);
    """))
    assert r["P_UNORD"]


# -- P_CLOCK -----------------------------------------------------------------

def test_clock_read_reaching_an_assertion():
    r = analyse(M("""
    long t = System.currentTimeMillis();
    assertTrue(t > 0);
    """))
    assert r["P_CLOCK"]


def test_instant_now_reaching_an_assertion():
    r = analyse(M("""
    Instant i = Instant.now();
    assertEquals(expected, i);
    """))
    assert r["P_CLOCK"]


def test_new_date_reaching_an_assertion():
    r = analyse(M("""
    Date d = new Date();
    assertNotNull(d);
    """))
    assert r["P_CLOCK"]


def test_clock_read_used_only_for_timing_is_not_a_hit():
    r = analyse(M("""
    long t0 = System.currentTimeMillis();
    doWork();
    long t1 = System.currentTimeMillis();
    assertTrue(ok);
    """))
    assert not r["P_CLOCK"]


# -- parsing -----------------------------------------------------------------

def test_parse_failure_is_reported_not_swallowed():
    r = analyse("public void t( {{{ ???")
    assert r["parse_ok"] is False
    assert all(r[p] is False for p in ("P_ASYNC", "P_UNORD", "P_CLOCK"))


def test_string_literals_do_not_trigger_properties():
    r = analyse(M('assertEquals("map.keySet()", s);'))
    assert not r["P_UNORD"]


def test_lambda_body_is_analysed():
    r = analyse(M("""
    pool.submit(() -> {
        counter.incrementAndGet();
    });
    assertEquals(1, counter.intValue());
    """))
    assert r["P_ASYNC"]


# -- vocabulary predicates ---------------------------------------------------

@pytest.mark.parametrize("name,expected", [
    ("assertTrue", True), ("assertThat", True), ("expectException", True),
    ("verify", True), ("fail", True), ("assertion", True),
    ("doWork", False), ("get", False), ("Assert", False),
])
def test_is_assertion(name, expected):
    assert is_assertion(name) is expected


@pytest.mark.parametrize("name,n,expected", [
    ("get", 0, True),            # Future.get()
    ("get", 2, True),            # Future.get(timeout, unit)
    ("get", 1, False),           # map.get(k)
    ("join", 0, True),           # Thread.join()
    ("join", 1, True),           # Thread.join(millis) -- a timed join still blocks
    ("join", 2, False),          # String.join(sep, parts)
    ("await", 0, True), ("await", 2, True), ("sleep", 1, True), ("submit", 1, False),
])
def test_is_sync(name, n, expected):
    assert is_sync(name, n) is expected


def test_zero_arg_get_on_an_unresolved_receiver_is_not_a_barrier():
    assert is_sync("get", 0, receiver_is_future=False) is False
    assert is_sync("get", 0, receiver_is_future=True) is True
    assert is_sync("sleep", 1, receiver_is_future=False) is True


def test_thread_join_is_a_barrier():
    r = analyse(M("""
    Thread t = new Thread(task);
    t.start();
    t.join();
    assertTrue(done);
    """))
    assert not r["P_ASYNC"]


def test_thread_start_without_join_reaches():
    r = analyse(M("""
    Thread t = new Thread(task);
    t.start();
    assertTrue(counter.get() > 0);
    """))
    assert r["P_ASYNC"]


def test_truncated_body_is_repaired_not_dropped():
    # FlakeBench cuts ~6% of method bodies off mid-statement
    truncated = ("@Test\npublic void t() {\n    fs.setAcl(target, Arrays.asList(\n"
                 "        aclEntry(ACCESS, USER, ALL),\n"
                 "        aclEntry(ACCESS, GROUP, NONE),")
    r = analyse(truncated)
    assert r["parse_ok"], r
    assert r["parse_repair"], r


def test_missing_closing_brace_is_repaired():
    r = analyse("@Test\npublic void t() {\n    executor.submit(task);\n    assertTrue(ok);")
    assert r["parse_ok"] and r["parse_repair"] == "close_brackets"
    assert r["P_ASYNC"]


def test_repair_never_invents_a_property():
    r = analyse("@Test\npublic void t() {\n    int x = 1;\n    doWork(")
    assert r["parse_ok"]
    assert not any(r[p] for p in ("P_ASYNC", "P_UNORD", "P_CLOCK"))


def test_sleep_kill_diagnostic_is_reported_separately():
    r = analyse(M("""
    executor.submit(task);
    Thread.sleep(100);
    assertTrue(done);
    """))
    assert not r["P_ASYNC"], "sleep is in the plan's kill set"
    assert r["P_ASYNC_sleep_kills"], "the diagnostic variant should still fire"


def test_static_factory_on_an_unordered_type_taints():
    r = analyse(M("""
    Multimap<Integer, String> inverse = HashMultimap.create();
    String json = JSON.toJSONString(inverse);
    assertEquals("{1:[\\"a\\"]}", json);
    """))
    assert r["P_UNORD"]


def test_guava_helper_factory_taints():
    r = analyse(M("""
    Set<String> s = Sets.newHashSet();
    assertEquals("[a, b]", s.toString());
    """))
    assert r["P_UNORD"]


def test_map_interface_with_opaque_initialiser_counts():
    r = analyse(M("""
    Map<Pair, Integer> weights = subject.buildWeights(pairs);
    assertEquals(expected, weights.toString());
    """))
    assert r["P_UNORD"]


def test_order_insensitive_accessor_does_not_count():
    r = analyse(M("""
    Map<String, Integer> m = build();
    assertEquals(2, m.size());
    assertTrue(m.containsKey("a"));
    """))
    assert not r["P_UNORD"], "size()/containsKey() do not depend on iteration order"


def test_asserting_on_the_collection_itself_does_not_count():
    r = analyse(M("""
    Set<String> s = Sets.newHashSet();
    assertEquals(expected, s);
    """))
    assert not r["P_UNORD"], "Set.equals is order-independent"


def test_map_interface_with_ordered_initialiser_does_not_count():
    r = analyse(M("""
    Map<String, Integer> m = new LinkedHashMap<>();
    assertEquals(expected, m);
    """))
    assert not r["P_UNORD"]


def test_list_declared_type_is_never_unordered():
    r = analyse(M("""
    List<String> l = build();
    assertEquals(expected, l);
    """))
    assert not r["P_UNORD"]


def test_timed_join_is_a_barrier():
    # surfaced by the hand audit: Thread.join(2000) blocks, so the dispatch does
    # not reach the assertion unsynchronised
    r = analyse(M("""
    Thread checker = new Thread(runnable);
    checker.start();
    checker.join(2000);
    assertTrue(conditionNotMet.get() == null);
    """))
    assert not r["P_ASYNC"]


def test_future_get_with_timeout_is_a_barrier():
    r = analyse(M("""
    Future<Integer> f = executor.submit(task);
    assertEquals(1, f.get(5, TimeUnit.SECONDS));
    """))
    assert not r["P_ASYNC"]


def test_string_join_is_not_a_barrier():
    r = analyse(M("""
    executor.submit(task);
    assertEquals("a,b", String.join(",", parts));
    """))
    assert r["P_ASYNC"]


# -- errors surfaced by the hand audit ---------------------------------------

def test_verify_chain_is_not_an_async_dispatch():
    # verify(mock).start() records an interaction; it does not start anything
    r = analyse(M("""
    latch.await();
    verify(repairSessions.get(range1)).start();
    verify(repairSessions.get(range2)).finish(eq(SUCCESS));
    """))
    assert not r["P_ASYNC"]


def test_real_start_still_counts_next_to_a_verify():
    r = analyse(M("""
    server.start();
    verify(mock).accept();
    """))
    assert r["P_ASYNC"]


def test_map_get_result_does_not_carry_collection_taint():
    # tokenData.get("k").get(0).getNode() does not depend on iteration order
    r = analyse(M("""
    Map<String, List<Detail>> tokenData = workflowToken.getTokenData();
    assertEquals(NAME, tokenData.get("start_time").get(0).getNode());
    """))
    assert not r["P_UNORD"]


def test_values_on_a_non_collection_receiver_does_not_count():
    # TestSubscriber.values() is a list of emitted items, not a Map view
    r = analyse(M("""
    assertThat(expectedChanges.containsAll(testSubscriber.values())).isTrue();
    """))
    assert not r["P_UNORD"]


def test_values_on_a_known_map_still_counts():
    r = analyse(M("""
    HashMap<String, Integer> m = new HashMap<>();
    assertEquals(expected, m.values().toString());
    """))
    assert r["P_UNORD"]


def test_try_with_resources_initialiser_is_analysed():
    # a `resource` node carries type/name/value directly, with no
    # variable_declarator child -- surfaced by the hand audit
    r = analyse(M("""
    try (Ignite ignored = Ignition.start(cfg)) {
        assertFalse(cache.replace(1, "2", "3"));
    }
    """))
    assert r["P_ASYNC"]


def test_try_with_resources_clock_resource():
    r = analyse(M("""
    try (Writer w = open(System.currentTimeMillis())) {
        assertNotNull(w);
    }
    """))
    assert r["P_CLOCK"]


# -- P_UNORD tightening: only actual iteration produces an ordered-dependent value

def test_collection_passed_to_an_opaque_call_does_not_taint_the_result():
    # the audit's five false positives all had this shape
    r = analyse(M("""
    final Map<String, Object> headers = new HashMap<>();
    final Parameters result = requestBodyAndHeaders("direct://ON_INSTANCE", null, headers);
    assertNotNull(result, "onInstance result");
    """))
    assert not r["P_UNORD"]


def test_quorum_over_a_map_is_not_order_dependent():
    r = analyse(M("""
    Map<String, Layout> layouts = new HashMap<>();
    Optional<Layout> quorumLayout = managementView.getLayoutFromQuorum(layouts, 2);
    assertThat(quorumLayout).isEqualTo(Optional.of(layout));
    """))
    assert not r["P_UNORD"]


def test_serialising_a_collection_still_taints():
    r = analyse(M("""
    Multimap<Integer, String> inverse = HashMultimap.create();
    String json = JSON.toJSONString(inverse);
    assertEquals("{}", json);
    """))
    assert r["P_UNORD"]


def test_iterating_a_set_still_taints():
    r = analyse(M("""
    Set<ValidationMessage> errors = schema.validate(node);
    for (ValidationMessage error : errors) {
        assertEquals(expected, error.getMessage());
    }
    """))
    assert r["P_UNORD"]


def test_keyset_iterator_still_taints():
    r = analyse(M("""
    assertEquals(channel.id(), selector.disconnected().keySet().iterator().next());
    """))
    assert r["P_UNORD"]
