# 04 -- hand-audit sample

50 methods, stratified by property and by label.

For each, decide by reading the code whether each property holds, using the
definitions in `plan/04_STRUCTURAL_LABELS.md`, then record the judgement in
`data/04_audit_labels.json` as `{"<id>": {"P_ASYNC": 0|1, ...}}`.

The analyser's own prediction is shown so the audit can be checked, not so it
can be copied. Judge from the code.

## uid 8dd700416785  (id 3, none&flaky)

- project: `eclipse_xtext-core`
- test: `RequestManagerTest.testRunWriteAfterRead`
- label: `concurrency`   parse_ok: 1
- analyser: {'P_ASYNC': 0, 'P_UNORD': 0, 'P_CLOCK': 0}
- witness: {}

```java
@Test
public void testRunWriteAfterRead() {
    final Function1<CancelIndicator, Integer> _function = (CancelIndicator it) -> {
        return Integer.valueOf(this.sharedState.incrementAndGet());
    };
    this.requestManager.<Integer>runRead(_function);
    final Function0<Object> _function_1 = () -> {
        return null;
    };
    final Function2<CancelIndicator, Object, Integer> _function_2 = (CancelIndicator $0,Object $1) -> {
        int _xblockexpression = ((int) (0));
        {
            Assert.assertEquals(1, this.sharedState.get());
            _xblockexpression = this.sharedState.incrementAndGet();
        }
        return Integer.valueOf(_xblockexpression);
    };
    this.requestManager.<Object, Integer>runWrite(_function_1, _function_2).join();
    Assert.assertEquals(2, this.sharedState.get());
}
```

## uid 5dd6208c06f0  (id 48, P_ASYNC=1&flaky)

- project: `cdapio_cdap`
- test: `PreviewDataPipelineTest.testLogicalTypePreviewRun`
- label: `time`   parse_ok: 1
- analyser: {'P_ASYNC': 1, 'P_UNORD': 0, 'P_CLOCK': 0}
- witness: {'P_ASYNC': 'Assert.assertEquals(actualRecordSamuel.get("date"), "2002-11-18")'}

```java
@Test
public void testLogicalTypePreviewRun(Engine engine) throws Exception {
    PreviewManager previewManager = getPreviewManager();
    String sourceTableName = "singleInput";
    String sinkTableName = "singleOutput";
    Schema schema = Schema.recordOf(
    "testRecord",
    Schema.Field.of("name", Schema.of(Schema.Type.STRING)),
    Schema.Field.of("date", Schema.of(Schema.LogicalType.DATE)),
    Schema.Field.of("ts", Schema.of(Schema.LogicalType.TIMESTAMP_MILLIS))
    );
    ETLBatchConfig etlConfig = ETLBatchConfig.builder()
    .addStage(new ETLStage("source", MockSource.getPlugin(sourceTableName, schema)))
    .addStage(new ETLStage("transform", IdentityTransform.getPlugin()))
    .addStage(new ETLStage("sink", MockSink.getPlugin(sinkTableName)))
    .addConnection("source", "transform")
    .addConnection("transform", "sink")
    .setEngine(engine)
    .setNumOfRecordsPreview(100)
    .build();
    PreviewConfig previewConfig = new PreviewConfig(SmartWorkflow.NAME, ProgramType.WORKFLOW,
    Collections.<String, String>emptyMap(), 10);
    addDatasetInstance(Table.class.getName(), sourceTableName,
    DatasetProperties.of(ImmutableMap.of("schema", schema.toString())));
    DataSetManager<Table> inputManager = getDataset(NamespaceId.DEFAULT.dataset(sourceTableName));
    ZonedDateTime expectedMillis = ZonedDateTime.of(2018, 11, 11, 11, 11, 11, 123 * 1000 * 1000,
    ZoneId.ofOffset("UTC", ZoneOffset.UTC));
    StructuredRecord recordSamuel = StructuredRecord.builder(schema).set("name", "samuel")
    .setDate("date", LocalDate.of(2002, 11, 18)).setTimestamp("ts", expectedMillis).build();
    StructuredRecord recordBob = StructuredRecord.builder(schema).set("name", "bob")
    .setDate("date", LocalDate.of(2003, 11, 18)).setTimestamp("ts", expectedMillis).build();
    MockSource.writeInput(inputManager, ImmutableList.of(recordSamuel, recordBob));
    AppRequest<ETLBatchConfig> appRequest = new AppRequest<>(APP_ARTIFACT_RANGE, etlConfig, previewConfig);
    ApplicationId previewId = previewManager.start(NamespaceId.DEFAULT, appRequest);
    Tasks.waitFor(PreviewStatus.Status.COMPLETED, new Callable<PreviewStatus.Status>() {
        @Override
        public PreviewStatus.Status call() throws Exception {
            PreviewStatus status = previewManager.getStatus(previewId);
            return status == null ? null : status.getStatus();
        }
    }, 5, TimeUnit.MINUTES);
    checkPreviewStore(previewManager, previewId, "source", 2);
    List<JsonElement> data = previewManager.getData(previewId, "source").get(DATA_TRACER_PROPERTY);
    StructuredRecord actualRecordSamuel = GSON.fromJson(data.get(0), StructuredRecord.class);
    Assert.assertEquals(actualRecordSamuel.get("date"), "2002-11-18");
    Assert.assertEquals(actualRecordSamuel.get("ts"), "2018-11-11T11:11:11.123Z[UTC]");
    StructuredRecord actualRecordBob = GSON.fromJson(data.get(1), StructuredRecord.class);
    Assert.assertEquals(actualRecordBob.get("date"), "2003-11-18");
    Assert.assertEquals(actualRecordBob.get("ts"), "2018-11-11T11:11:11.123Z[UTC]");
    checkPreviewStore(previewManager, previewId, "transform", 2);
    checkPreviewStore(previewManager, previewId, "sink", 2);
    validateMetric(2, previewId, "source.records.in", previewManager);
    validateMetric(2, previewId, "source.records.out", previewManager);
    validateMetric(2, previewId, "transform.records.in", previewManager);
    validateMetric(2, previewId, "transform.records.out", previewManager);
    validateMetric(2, previewId, "sink.records.out", previewManager);
    validateMetric(2, previewId, "sink.records.in", previewManager);
    DataSetManager<Table> sinkManager = getDataset(sinkTableName);
    Assert.assertNull(sinkManager.get());
    deleteDatasetInstance(NamespaceId.DEFAULT.dataset(sourceTableName));
    Assert.assertNotNull(previewManager.getRunId(previewId));
}
```

## uid 5a0353688b31  (id 63, none&flaky)

- project: `wildfly_wildfly`
- test: `b19048b72669fc0e96665b1b125dc1fda21f5993.testLookupWithContinuation`
- label: `test order dependency`   parse_ok: 1
- analyser: {'P_ASYNC': 0, 'P_UNORD': 0, 'P_CLOCK': 0}
- witness: {}

```java
@Test
public void testLookupWithContinuation() throws Exception {
    namingStore.bind(new CompositeName("comp/nested"), "test");
    final Reference reference = new Reference(String.class.getName(), new StringRefAddr("nns", "comp"), TestObjectFactoryWithNameResolution.class.getName(), null);
    namingStore.bind(new CompositeName("test"), reference);
    Object result = namingContext.lookup(new CompositeName("test/nested"));
    assertEquals("test", result);
    result = testActionPermission(JndiPermission.ACTION_LOOKUP, Arrays.asList(new JndiPermission("comp/nested", "lookup")), namingContext, "test/nested");
    assertEquals("test", result);
}
```

## uid f609c28e2949  (id 71, P_CLOCK=1&flaky)

- project: `androidx_androidx`
- test: `testOneTimeWorkRequest_backedOff`
- label: `time`   parse_ok: 0
- analyser: {'P_ASYNC': 0, 'P_UNORD': 0, 'P_CLOCK': 1}
- witness: {'P_CLOCK': 'assertEquals(task.windowStart, offset)'}

```java
@Test
public void testOneTimeWorkRequest_backedOff() {
    val now = System.currentTimeMillis() ;
    when(mTaskConverter.now()).thenReturn(now) ;
    val request = OneTimeWorkRequestBuilder<TestWorker>().setInitialRunAttemptCount(1).build() ;
    val workSpec = request.workSpec ;
    val expected = workSpec.calculateNextRunTime();
    val offset = offset(expected, now) ,
    val delta = task.windowEnd - (offset + EXECUTION_WINDOW_SIZE_IN_SECONDS);
    val task = mTaskConverter.convert(request.workSpec);
    assertEquals(task.serviceName, WorkManagerGcmService::class.java.name);
    assertEquals(task.isPersisted, false);
    assertEquals(task.isUpdateCurrent, true);
    assertEquals(task.requiredNetwork, Task.NETWORK_STATE_ANY);
    assertEquals(task.requiresCharging, false);
    assertEquals(task.windowStart, offset);
    assertEquals(task.windowEnd, offset + EXECUTION_WINDOW_SIZE_IN_SECONDS);
}
```

## uid 7948d927f4f1  (id 101, P_CLOCK=1&flaky)

- project: `apache_avro`
- test: `testRecordWithJsr310LogicalTypes`
- label: `time`   parse_ok: 1
- analyser: {'P_ASYNC': 0, 'P_UNORD': 0, 'P_CLOCK': 1}
- witness: {'P_CLOCK': 'Assert.assertEquals("Should match written record", record, actual.get(0))'}

```java
@Test
public void testRecordWithJsr310LogicalTypes() throws IOException {
    TestRecordWithJsr310LogicalTypes record = new TestRecordWithJsr310LogicalTypes(
    true,
    34,
    35L,
    3.14F,
    3019.34,
    null,
    java.time.LocalDate.now(),
    java.time.LocalTime.now().truncatedTo(ChronoUnit.MILLIS),
    java.time.Instant.now().truncatedTo(ChronoUnit.MILLIS),
    new BigDecimal(123.45f).setScale(2, BigDecimal.ROUND_HALF_DOWN)
    );
    File data = write(TestRecordWithJsr310LogicalTypes.getClassSchema(), record);
    List<TestRecordWithJsr310LogicalTypes> actual = read(
    TestRecordWithJsr310LogicalTypes.getClassSchema(), data);
    Assert.assertEquals("Should match written record", record, actual.get(0));
}
```

## uid d49cd59a7cd6  (id 146, P_CLOCK=1&flaky)

- project: `Alluxio_alluxio`
- test: `FileSystemMasterIntegrationTest.lastModificationTimeAddCheckpointTest`
- label: `time`   parse_ok: 1
- analyser: {'P_ASYNC': 0, 'P_UNORD': 0, 'P_CLOCK': 1}
- witness: {'P_CLOCK': 'Assert.assertEquals(opTimeMs, fileInfo.lastModificationTimeMs)'}

```java
@Test
public void lastModificationTimeAddCheckpointTest() throws Exception {
    long fileId = mFsMaster.create(new TachyonURI("/testFile"), CreateOptions.defaults());
    long opTimeMs = System.currentTimeMillis();
    mFsMaster.persistFileInternal(fileId, 1, opTimeMs);
    FileInfo fileInfo = mFsMaster.getFileInfo(fileId);
    Assert.assertEquals(opTimeMs, fileInfo.lastModificationTimeMs);
}
```

## uid 36bbe2787fb4  (id 191, none&flaky)

- project: `apache_pulsar`
- test: `DiscoveryServiceTest.testBrokerDiscoveryRoundRobin`
- label: `async wait`   parse_ok: 1
- analyser: {'P_ASYNC': 0, 'P_UNORD': 0, 'P_CLOCK': 0}
- witness: {}

```java
@Test
public void testBrokerDiscoveryRoundRobin() throws Exception {
    addBrokerToZk(5);
    String prevUrl = null;
    for (int i = 0; i < 10; i++) {
        String current = service.getDiscoveryProvider().nextBroker().getPulsarServiceUrl();
        assertNotEquals(prevUrl, current);
        prevUrl = current;
    }
}
```

## uid 4c061ec7229b  (id 206, P_UNORD=1&flaky)

- project: `apache_kafka`
- test: `testGracefulClose`
- label: `async wait`   parse_ok: 1
- analyser: {'P_ASYNC': 0, 'P_UNORD': 1, 'P_CLOCK': 0}
- witness: {'P_UNORD': 'assertEquals(channel.id(), selector.disconnected().keySet().iterator().next())'}

```java
@Test
public void testGracefulClose() throws Exception {
    int maxReceiveCountAfterClose = 0;
    for (int i = 6; i <= 100 && maxReceiveCountAfterClose < 5; i++) {
        int receiveCount = 0;
        KafkaChannel channel = createConnectionWithPendingReceives(i);
        selector.poll(1000);
        assertEquals(1, selector.completedReceives().size());
        server.closeConnections();
        while (selector.disconnected().isEmpty()) {
            selector.poll(1);
            receiveCount += selector.completedReceives().size();
            assertTrue("Too many completed receives in one poll", selector.completedReceives().size() <= 1);
        }
        assertEquals(channel.id(), selector.disconnected().keySet().iterator().next());
        maxReceiveCountAfterClose = Math.max(maxReceiveCountAfterClose, receiveCount);
    }
    assertTrue("Too few receives after close: " + maxReceiveCountAfterClose, maxReceiveCountAfterClose >= 5);
}
```

## uid 1f6890e85f8e  (id 212, P_ASYNC=1&flaky)

- project: `square_okhttp`
- test: `HttpOverHttp2Test.recoverFromCancelReusesConnection`
- label: `async wait`   parse_ok: 1
- analyser: {'P_ASYNC': 1, 'P_UNORD': 0, 'P_CLOCK': 0}
- witness: {'P_ASYNC': 'assertThat(response.body().string())'}

```java
@Test
public void recoverFromCancelReusesConnection() throws Exception {
    CountDownLatch responseDequeuedLatch = new CountDownLatch(1);
    CountDownLatch requestCanceledLatch = new CountDownLatch(1);
    QueueDispatcher dispatcher = new QueueDispatcher() {
        @Override
        public MockResponse dispatch(RecordedRequest request) throws InterruptedException {
            MockResponse response = super.dispatch(request);
            responseDequeuedLatch.countDown();
            requestCanceledLatch.await();
            return response;
        }
    };
    server.setDispatcher(dispatcher);
    dispatcher.enqueueResponse(new MockResponse().setBodyDelay(10, TimeUnit.SECONDS).setBody("abc"));
    dispatcher.enqueueResponse(new MockResponse().setBody("def"));
    client = client.newBuilder().dns(new DoubleInetAddressDns()).build();
    callAndCancel(0, responseDequeuedLatch, requestCanceledLatch);
    Call call = client.newCall(new Request.Builder().url(server.url("/")).build());
    Response response = call.execute();
    assertThat(response.body().string()).isEqualTo("def");
    assertThat(server.takeRequest().getSequenceNumber()).isEqualTo(1);
}
```

## uid c96d693569db  (id 215, P_ASYNC=1&flaky)

- project: `neo4j_neo4j`
- test: `shouldPickANewServerToWriteToOnLeaderSwitch`
- label: `concurrency`   parse_ok: 1
- analyser: {'P_ASYNC': 1, 'P_UNORD': 0, 'P_CLOCK': 0}
- witness: {'P_ASYNC': 'fail( "Failed to write to the new leader in time" )'}

```java
@Test
public void shouldPickANewServerToWriteToOnLeaderSwitch() throws Throwable
{
    cluster = clusterRule.withNumberOfEdgeMembers( 0 ).startCluster();
    CoreClusterMember leader = cluster.awaitLeader();
    CountDownLatch startTheLeaderSwitching = new CountDownLatch( 1 );
    Thread thread = new Thread( () ->
    {
        try
        {
            startTheLeaderSwitching.await();
            CoreClusterMember theLeader = cluster.awaitLeader();
            switchLeader( theLeader );
        }
        catch ( TimeoutException | InterruptedException e )
        {
        }
    } );
    thread.start();
    Config config = Config.build().withLogging( new JULogging( Level.OFF ) ).toConfig();
    try ( Driver driver = GraphDatabase
    .driver( leader.routingURI(), AuthTokens.basic( "neo4j", "neo4j" ), config ) )
    {
        boolean success = false;
        Set<BoltServerAddress> seenAddresses = new HashSet<>();
        long deadline = System.currentTimeMillis() + (30 * 1000);
        while ( !success )
        {
            if ( System.currentTimeMillis() > deadline )
            {
                fail( "Failed to write to the new leader in time" );
            }
            try ( Session session = driver.session( AccessMode.WRITE ) )
            {
                startTheLeaderSwitching.countDown();
                BoltServerAddress boltServerAddress = ((RoutingNetworkSession) session).address();
                seenAddresses.add( boltServerAddress );
                session.run( "CREATE (p:Person)" );
                success = seenAddresses.size() >= 2;
            }
            catch ( Exception e )
            {
                Thread.sleep( 100 );
            }
        }
    }
    finally
    {
        thread.join();
    }
}
```

## uid b903d0f2a511  (id 244, P_CLOCK=1&flaky)

- project: `kiwiproject_dropwizard-service-utilities`
- test: `SystemExecutionerTest.shouldExitBeforeGivenWaitTime_WhenWaitingThreadInterrupted`
- label: `time`   parse_ok: 1
- analyser: {'P_ASYNC': 0, 'P_UNORD': 0, 'P_CLOCK': 1}
- witness: {'P_CLOCK': 'assertThat(TimeUnit.NANOSECONDS.toMillis(elapsedNanos))'}

```java
@Test
void shouldExitBeforeGivenWaitTime_WhenWaitingThreadInterrupted() {
    var executorService = Executors.newFixedThreadPool(2);
    var executionStrategy = new ExecutionStrategies.ExitFlaggingExecutionStrategy();
    var executioner = new SystemExecutioner(executionStrategy);
    var startTime = new AtomicLong();
    var executionFuture = executorService.submit(() -> {
        LOG.info("Calling executioner with 5 second wait");
        startTime.set(System.nanoTime());
        executioner.exit(5, TimeUnit.SECONDS);
    });
    var killerSleepTimeMillis = 100;
    var killerFuture = executorService.submit(() -> {
        LOG.info("Sleeping for {} milliseconds...", killerSleepTimeMillis);
        new DefaultEnvironment().sleepQuietly(killerSleepTimeMillis, TimeUnit.MILLISECONDS);
        LOG.info("I'm awake and will now interrupt executionThread");
        var canceled = executionFuture.cancel(true);
        LOG.info("executionFuture was canceled? {}", canceled);
    });
    await().atMost(ONE_SECOND).until(() -> executionFuture.isDone() && killerFuture.isDone());
    long elapsedNanos = System.nanoTime() - startTime.get();
    assertThat(executionStrategy.didExit()).describedAs("Execution strategy exit() should have been called").isTrue();
    assertThat(TimeUnit.NANOSECONDS.toMillis(elapsedNanos)).describedAs("Elapsed millis must be greater than %d", killerSleepTimeMillis).isGreaterThan(killerSleepTimeMillis);
    executorService.shutdown();
    await().atMost(ONE_SECOND).until(executorService::isShutdown);
}
```

## uid 98f9550c7dc6  (id 258, P_ASYNC=1&flaky)

- project: `apache_hadoop`
- test: `TestMetricsSystemImpl.testInitFirstVerifyCallBacks`
- label: `unordered collections`   parse_ok: 1
- analyser: {'P_ASYNC': 1, 'P_UNORD': 0, 'P_CLOCK': 0}
- witness: {'P_ASYNC': 'verify(sink1, timeout(200).times(2))'}

```java
@Test
public void testInitFirstVerifyCallBacks() throws Exception {
    DefaultMetricsSystem.shutdown();
    new ConfigBuilder().add("*.period", 8).add("test.sink.test.class", TestSink.class.getName()).add("test.*.source.filter.exclude", "s0").add("test.source.s1.metric.filter.exclude", "X*").add("test.sink.sink1.metric.filter.exclude", "Y*").add("test.sink.sink2.metric.filter.exclude", "Y*").save(TestMetricsConfig.getTestFilename("hadoop-metrics2-test"));
    MetricsSystemImpl ms = new MetricsSystemImpl("Test");
    ms.start();
    ms.register("s0", "s0 desc", new TestSource("s0rec"));
    TestSource s1 = ms.register("s1", "s1 desc", new TestSource("s1rec"));
    s1.c1.incr();
    s1.xxx.incr();
    s1.g1.set(2);
    s1.yyy.incr(2);
    s1.s1.add(0);
    MetricsSink sink1 = mock(MetricsSink.class);
    MetricsSink sink2 = mock(MetricsSink.class);
    ms.registerSink("sink1", "sink1 desc", sink1);
    ms.registerSink("sink2", "sink2 desc", sink2);
    ms.publishMetricsNow();
    try {
        verify(sink1, timeout(200).times(2)).putMetrics(r1.capture());
        verify(sink2, timeout(200).times(2)).putMetrics(r2.capture());
    } finally {
        ms.stop();
        ms.shutdown();
    }
    List<MetricsRecord> mr1 = r1.getAllValues();
    List<MetricsRecord> mr2 = r2.getAllValues();
    checkMetricsRecords(mr1);
    assertEquals("output", mr1, mr2);
}
```

## uid e97b7bbcbf24  (id 262, none&flaky)

- project: `cdapio_cdap`
- test: `WorkflowHttpHandlerTest.testWorkflowTokenPut`
- label: `async wait`   parse_ok: 1
- analyser: {'P_ASYNC': 0, 'P_UNORD': 0, 'P_CLOCK': 0}
- witness: {}

```java
@Test
public void testWorkflowTokenPut() throws Exception {
    Assert.assertEquals(200, deploy(WorkflowTokenTestPutApp.class).getStatusLine().getStatusCode());
    Id.Application appId = Id.Application.from(Id.Namespace.DEFAULT, WorkflowTokenTestPutApp.NAME);
    Id.Workflow workflowId = Id.Workflow.from(appId, WorkflowTokenTestPutApp.WorkflowTokenTestPut.NAME);
    Id.Program mapReduceId = Id.Program.from(appId, ProgramType.MAPREDUCE, WorkflowTokenTestPutApp.RecordCounter.NAME);
    Id.Program sparkId = Id.Program.from(appId, ProgramType.SPARK, WorkflowTokenTestPutApp.SparkTestApp.NAME);
    String outputPath = new File(tmpFolder.newFolder(), "output").getAbsolutePath();
    startProgram(workflowId, ImmutableMap.of("inputPath", createInputForRecordVerification("firstInput"),
    "outputPath", outputPath, "put.in.mapper.initialize", "true"));
    waitState(workflowId, ProgramRunStatus.RUNNING.name());
    waitState(workflowId, "STOPPED");
    List<RunRecord> workflowProgramRuns = getProgramRuns(workflowId, ProgramRunStatus.FAILED.name());
    Assert.assertEquals(1, workflowProgramRuns.size());
    List<RunRecord> mapReduceProgramRuns = getProgramRuns(mapReduceId, ProgramRunStatus.FAILED.name());
    Assert.assertEquals(1, mapReduceProgramRuns.size());
    outputPath = new File(tmpFolder.newFolder(), "output").getAbsolutePath();
    startProgram(workflowId, ImmutableMap.of("inputPath", createInputForRecordVerification("secondInput"),
    "outputPath", outputPath, "put.in.map", "true"));
    waitState(workflowId, ProgramRunStatus.RUNNING.name());
    waitState(workflowId, "STOPPED");
    workflowProgramRuns = getProgramRuns(workflowId, ProgramRunStatus.FAILED.name());
    Assert.assertEquals(2, workflowProgramRuns.size());
    mapReduceProgramRuns = getProgramRuns(mapReduceId, ProgramRunStatus.FAILED.name());
    Assert.assertEquals(2, mapReduceProgramRuns.size());
    outputPath = new File(tmpFolder.newFolder(), "output").getAbsolutePath();
    startProgram(workflowId, ImmutableMap.of("inputPath", createInputForRecordVerification("thirdInput"),
    "outputPath", outputPath, "put.in.reducer.initialize", "true"));
    waitState(workflowId, ProgramRunStatus.RUNNING.name());
    waitState(workflowId, "STOPPED");
    workflowProgramRuns = getProgramRuns(workflowId, ProgramRunStatus.FAILED.name());
    Assert.assertEquals(3, workflowProgramRuns.size());
    mapReduceProgramRuns = getProgramRuns(mapReduceId, ProgramRunStatus.FAILED.name());
    Assert.assertEquals(3, mapReduceProgramRuns.size());
    outputPath = new File(tmpFolder.newFolder(), "output").getAbsolutePath();
    startProgram(workflowId, ImmutableMap.of("inputPath", createInputForRecordVerification("fourthInput"),
    "outputPath", outputPath, "put.in.reduce", "true"));
    waitState(workflowId, ProgramRunStatus.RUNNING.name());
    waitState(workflowId, "STOPPED");
    workflowProgramRuns = getProgramRuns(workflowId, ProgramRunStatus.FAILED.name());
    Assert.assertEquals(4, workflowProgramRuns.size());
    mapReduceProgramRuns = getProgramRuns(mapReduceId, ProgramRunStatus.FAILED.name());
    Assert.assertEquals(4, mapReduceProgramRuns.size());
    outputPath = new File(tmpFolder.newFolder(), "output").getAbsolutePath();
    startProgram(workflowId, ImmutableMap.of("inputPath", createInputForRecordVerification("fifthInput"),
    "outputPath", outputPath, "closurePutToken", "true"));
    waitState(workflowId, ProgramRunStatus.RUNNING.name());
    waitState(workflowId, "STOPPED");
    workflowProgramRuns = getProgramRuns(workflowId, ProgramRunStatus.FAILED.name());
    Assert.assertEquals(5, workflowProgramRuns.size());
    mapReduceProgramRuns = getProgramRuns(mapReduceId, ProgramRunStatus.COMPLETED.name());
    Assert.assertEquals(1, mapReduceProgramRuns.size());
    List<RunRecord> sparkProgramRuns = getProgramRuns(sparkId, ProgramRunStatus.FAILED.name());
    Assert.assertEquals(1, sparkProgramRuns.size());
    outputPath = new File(tmpFolder.newFolder(), "output").getAbsolutePath();
    startProgram(workflowId, ImmutableMap.of("inputPath", createInputForRecordVerification("sixthInput"),
    "outputPath", outputPath));
    waitState(workflowId, ProgramRunStatus.RUNNING.name());
    waitState(workflowId, "STOPPED");
    workflowProgramRuns = getProgramRuns(workflowId, ProgramRunStatus.COMPLETED.name());
    Assert.assertEquals(1, workflowProgramRuns.size());
    workflowProgramRuns = getProgramRuns(sparkId, ProgramRunStatus.COMPLETED.name());
    Assert.assertEquals(1, workflowProgramRuns.size());
}
```

## uid 08193d0a60ef  (id 276, none&flaky)

- project: `tbsalling_aismessages`
- test: `7b0c4c708b6bb9a6da3d5737bcad1857ade8a931.canHandleUnfragmentedMessageReceived`
- label: `test order dependency`   parse_ok: 1
- analyser: {'P_ASYNC': 0, 'P_UNORD': 0, 'P_CLOCK': 0}
- witness: {}

```java
@Test
public void canHandleUnfragmentedMessageReceived() {
    NMEAMessage unfragmentedNMEAMessage = NMEAMessage.fromString("!AIVDM,1,1,,B,15MqdBP000G@qoLEi69PVGaN0D0=,0*3A");
    final ArgumentCaptor<AISMessage> aisMessage = new ArgumentCaptor<>();
    context.checking(new Expectations() {{
        oneOf(aisMessageHandler).accept(with(aisMessage.getMatcher()));
    }});
    aisMessageReceiver.accept(unfragmentedNMEAMessage);
    assertEquals(AISMessageType.PositionReportClassAScheduled, aisMessage.getCapturedObject().getMessageType());
}
```

## uid 3cd8087b7c0a  (id 310, P_CLOCK=1&flaky)

- project: `apache_cassandra`
- test: `testTrackMetadata_rowMarkerDelete`
- label: `time`   parse_ok: 1
- analyser: {'P_ASYNC': 1, 'P_UNORD': 0, 'P_CLOCK': 1}
- witness: {'P_ASYNC': 'assertEquals(1, cfs.getLiveSSTables().size())', 'P_CLOCK': 'assertEquals(System.currentTimeMillis()/1000, metadata.maxLocalDeletionTime, 5)'}

```java
@Test
public void testTrackMetadata_rowMarkerDelete() throws Throwable
{
    createTable("CREATE TABLE %s (a int, PRIMARY KEY (a))");
    ColumnFamilyStore cfs = Keyspace.open(keyspace()).getColumnFamilyStore(currentTable());
    execute("DELETE FROM %s USING TIMESTAMP 9999 WHERE a=1");
    cfs.forceBlockingFlush();
    assertEquals(1, cfs.getLiveSSTables().size());
    StatsMetadata metadata = cfs.getLiveSSTables().iterator().next().getSSTableMetadata();
    assertEquals(9999, metadata.minTimestamp);
    assertEquals(9999, metadata.maxTimestamp);
    assertEquals(System.currentTimeMillis()/1000, metadata.maxLocalDeletionTime, 5);
    cfs.forceMajorCompaction();
    StatsMetadata metadata2 = cfs.getLiveSSTables().iterator().next().getSSTableMetadata();
    assertEquals(metadata.maxLocalDeletionTime, metadata2.maxLocalDeletionTime);
    assertEquals(metadata.minTimestamp, metadata2.minTimestamp);
    assertEquals(metadata.maxTimestamp, metadata2.maxTimestamp);
}
```

## uid f2afe5d6289f  (id 325, P_CLOCK=1&flaky)

- project: `cdapio_cdap`
- test: `MetadataHttpHandlerTestRun.testSystemMetadataRetrieval`
- label: `async wait`   parse_ok: 1
- analyser: {'P_ASYNC': 0, 'P_UNORD': 0, 'P_CLOCK': 1}
- witness: {'P_CLOCK': 'Assert.assertTrue("Stream create time should be within the last hour - " + createTime, createTime > (System.currentTimeM'}

```java
@Test
public void testSystemMetadataRetrieval() throws Exception {
    appClient.deploy(DEFAULT, createAppJarFile(AllProgramsApp.class));
    Id.Stream streamId = Stream.from(DEFAULT, STREAM_NAME);
    Set<String> streamSystemTags = getTags(streamId, SYSTEM);
    Assert.assertEquals(ImmutableSet.of(STREAM_NAME), streamSystemTags);
    Map<String, String> streamSystemProperties = getProperties(streamId, SYSTEM);
    final String creationTime = "creation-time";
    String description = "description";
    String schema = "schema";
    String ttl = "ttl";
    Assert.assertTrue("Expected creation time to exist but it does not", streamSystemProperties.containsKey(creationTime));
    long createTime = Long.parseLong(streamSystemProperties.get(creationTime));
    Assert.assertTrue("Stream create time should be within the last hour - " + createTime, createTime > (System.currentTimeMillis() - TimeUnit.HOURS.toMillis(1)));
    Assert.assertEquals(ImmutableMap.of(schema, Schema.recordOf("stringBody", Field.of("body", Schema.of(STRING))).toString(), ttl, String.valueOf(Long.MAX_VALUE), description, "test stream", creationTime, String.valueOf(createTime)), streamSystemProperties);
    long newTtl = 100000L;
    streamClient.setStreamProperties(streamId, new StreamProperties(newTtl, null, null));
    streamSystemProperties = getProperties(streamId, SYSTEM);
    Assert.assertEquals(ImmutableMap.of(schema, Schema.recordOf("stringBody", Field.of("body", Schema.of(STRING))).toString(), ttl, String.valueOf(newTtl * 1000), description, "test stream", creationTime, String.valueOf(createTime)), streamSystemProperties);
    Set<MetadataRecord> streamSystemMetadata = getMetadata(streamId, SYSTEM);
    Assert.assertEquals(ImmutableSet.of(new MetadataRecord(streamId, MetadataScope.SYSTEM, streamSystemProperties, streamSystemTags)), streamSystemMetadata);
    Id.Stream.View view = View.from(streamId, "view");
    Schema viewSchema = Schema.recordOf("record", Field.of("viewBody", Schema.nullableOf(Schema.of(BYTES))));
    streamViewClient.createOrUpdate(view, new ViewSpecification(new FormatSpecification("format", viewSchema)));
    Set<String> viewSystemTags = getTags(view, SYSTEM);
    Assert.assertEquals(ImmutableSet.of("view", STREAM_NAME), viewSystemTags);
    Map<String, String> viewSystemProperties = getProperties(view, SYSTEM);
    Assert.assertEquals(viewSchema.toString(), viewSystemProperties.get(schema));
    ImmutableSet<String> viewUserTags = ImmutableSet.of("viewTag");
    addTags(view, viewUserTags);
    Assert.assertEquals(ImmutableSet.of(new MetadataRecord(view, MetadataScope.USER, ImmutableMap.<String, String>of(), viewUserTags), new MetadataRecord(view, MetadataScope.SYSTEM, viewSystemProperties, viewSystemTags)), getMetadata(view));
    Id.DatasetInstance datasetInstance = DatasetInstance.from(DEFAULT, DATASET_NAME);
    Set<String> dsSystemTags = getTags(datasetInstance, SYSTEM);
    Assert.assertEquals(ImmutableSet.of(DATASET_NAME, BATCH_TAG, EXPLORE_TAG), dsSystemTags);
    Map<String, String> dsSystemProperties = getProperties(datasetInstance, SYSTEM);
    Assert.assertTrue("Expected creation time to exist but it does not", dsSystemProperties.containsKey(creationTime));
    createTime = Long.parseLong(dsSystemProperties.get(creationTime));
    Assert.assertTrue("Dataset create time should be within the last hour - " + createTime, createTime > (System.currentTimeMillis() - TimeUnit.HOURS.toMillis(1)));
    Assert.assertEquals(ImmutableMap.of("type", KeyValueTable.class.getName(), description, "test dataset", creationTime, String.valueOf(createTime)), dsSystemProperties);
    datasetClient.update(datasetInstance, ImmutableMap.of(PROPERTY_TTL, "100000"));
    dsSystemProperties = getProperties(datasetInstance, SYSTEM);
    Assert.assertEquals(ImmutableMap.of("type", KeyValueTable.class.getName(), description, "test dataset", ttl, "100000", creationTime, String.valueOf(createTime)), dsSystemProperties);
    Id.Artifact artifactId = getArtifactId();
    Assert.assertEquals(ImmutableSet.of(new MetadataRecord(artifactId, MetadataScope.SYSTEM, ImmutableMap.<String, String>of(), ImmutableSet.of(AllProgramsApp.class.getSimpleName()))), getMetadata(artifactId, SYSTEM));
    Id.Application app = Application.from(DEFAULT, NAME);
    Assert.assertEquals(ImmutableMap.builder().put((FLOW.getPrettyName() + MetadataDataset.KEYVALUE_SEPARATOR) + NoOpFlow.NAME, NAME).put((MAPREDUCE.getPrettyName() + MetadataDataset.KEYVALUE_SEPARATOR) + NoOpMR.NAME, NAME).put((MAPREDUCE.getPrettyName() + MetadataDataset.KEYVALUE_SEPARATOR) + NoOpMR2.NAME, NAME).put((SERVICE.getPrettyName() + MetadataDataset.KEYVALUE_SEPARATOR) + NoOpService.NAME, NAME).put((SPARK.getPrettyName() + MetadataDataset.KEYVALUE_SEPARATOR) + NoOpSpark.NAME, NAME).put((WORKER.getPrettyName() + MetadataDataset.KEYVALUE_SEPARATOR) + NoOpWorker.NAME, NAME).put((WORKFLOW.getPrettyName() + MetadataDataset.KEYVALUE_SEPARATOR) + NoOpWorkflow.NAME, NAME).put(("schedule" + MetadataDataset.KEYVALUE_SEPARATOR) + AllProgramsApp.SCHEDULE_NAME, (AllProgramsApp.SCHEDULE_NAME + MetadataDataset.KEYVALUE_SEPARATOR) + AllProgramsApp.SCHEDULE_DESCRIPTION).build(), getProperties(app, SYSTEM));
    Assert.assertEquals(ImmutableSet.of(AllProgramsApp.class.getSimpleName(), NAME), getTags(app, SYSTEM));
    assertProgramSystemMetadata(Program.from(app, FLOW, NAME), "Realtime");
    assertProgramSystemMetadata(Program.from(app, WORKER, NAME), "Realtime");
    assertProgramSystemMetadata(Program.from(app, SERVICE, NAME), "Realtime");
    assertProgramSystemMetadata(Program.from(app, MAPREDUCE, NAME), "Batch");
    assertProgramSystemMetadata(Program.from(app, SPARK, NAME), "Batch");
    assertProgramSystemMetadata(Program.from(app, WORKFLOW, NAME), "Batch");
}
```

## uid d8e6f9de8342  (id 330, none&flaky)

- project: `androidx_androidx`
- test: `invalidationInAnotherInstance_closed`
- label: `async wait`   parse_ok: 1
- analyser: {'P_ASYNC': 0, 'P_UNORD': 0, 'P_CLOCK': 0}
- witness: {}

```java
@Test
public void invalidationInAnotherInstance_closed() throws Exception {
    final SampleDatabase db1 = openDatabase(true);
    final SampleDatabase db2 = openDatabase(true);
    final SampleDatabase db3 = openDatabase(true);
    final CountDownLatch invalidated1 = prepareTableObserver(db1);
    final Pair<CountDownLatch, CountDownLatch> changed1 = prepareLiveDataObserver(db1);
    final CountDownLatch invalidated2 = prepareTableObserver(db2);
    final Pair<CountDownLatch, CountDownLatch> changed2 = prepareLiveDataObserver(db2);
    final CountDownLatch invalidated3 = prepareTableObserver(db3);
    final Pair<CountDownLatch, CountDownLatch> changed3 = prepareLiveDataObserver(db3);
    db2.getCustomerDao().insert(CUSTOMER_1);
    assertTrue(invalidated1.await(3, TimeUnit.SECONDS));
    assertTrue(changed1.first.await(3, TimeUnit.SECONDS));
    assertTrue(invalidated2.await(3, TimeUnit.SECONDS));
    assertTrue(changed2.first.await(3, TimeUnit.SECONDS));
    assertTrue(invalidated3.await(3, TimeUnit.SECONDS));
    assertTrue(changed3.first.await(3, TimeUnit.SECONDS));
    db3.close();
    db2.getCustomerDao().insert(CUSTOMER_2);
    assertTrue(changed1.second.await(3, TimeUnit.SECONDS));
    assertTrue(changed2.second.await(3, TimeUnit.SECONDS));
    assertFalse(changed3.second.await(300, TimeUnit.MILLISECONDS));
}
```

## uid 43bb65c3cc25  (id 332, P_ASYNC=1&flaky)

- project: `ConsenSys_teku`
- test: `SyncCommitteeGossipAcceptanceTest.shouldContainSyncCommitteeAggregates`
- label: `async wait`   parse_ok: 1
- analyser: {'P_ASYNC': 1, 'P_UNORD': 0, 'P_CLOCK': 0}
- witness: {'P_ASYNC': 'assertThat(primaryNode.getContributionAndProofEvents().stream().filter(( proof) -> proof.message.aggregatorIndex.isGreat'}

```java
@Test
public void shouldContainSyncCommitteeAggregates() throws Exception {
    primaryNode.start();
    primaryNode.startEventListener(List.of(contribution_and_proof));
    secondaryNode.start();
    secondaryNode.startEventListener(List.of(contribution_and_proof));
    validatorClient.start();
    primaryNode.waitForEpoch(1);
    secondaryNode.waitForFullSyncCommitteeAggregate();
    validatorClient.stop();
    secondaryNode.stop();
    primaryNode.stop();
    assertThat(primaryNode.getContributionAndProofEvents().stream().filter(( proof) -> proof.message.aggregatorIndex.isGreaterThanOrEqualTo(8)).count()).isGreaterThan(0);
    assertThat(secondaryNode.getContributionAndProofEvents().stream().filter(( proof) -> proof.message.aggregatorIndex.isLessThan(8)).count()).isGreaterThan(0);
}
```

## uid f147040cf6fe  (id 342, P_ASYNC=1&flaky)

- project: `cdapio_cdap`
- test: `TestFrameworkTestRun.testAppWithServices`
- label: `concurrency`   parse_ok: 1
- analyser: {'P_ASYNC': 1, 'P_UNORD': 0, 'P_CLOCK': 0}
- witness: {'P_ASYNC': 'Assert.assertNotNull(serviceURL)'}

```java
@Test
public void testAppWithServices() throws Exception {
    ApplicationManager applicationManager = deployApplication(AppWithServices.class);
    LOG.info("Deployed.");
    ServiceManager serviceManager = applicationManager.getServiceManager(AppWithServices.SERVICE_NAME).start();
    serviceManager.waitForStatus(true);
    LOG.info("Service Started");
    URL serviceURL = serviceManager.getServiceURL(15, TimeUnit.SECONDS);
    Assert.assertNotNull(serviceURL);
    URL url = new URL(serviceURL, "ping2");
    HttpRequest request = HttpRequest.get(url).build();
    HttpResponse response = HttpRequests.execute(request);
    Assert.assertEquals(200, response.getResponseCode());
    url = new URL(serviceURL, "failure");
    request = HttpRequest.get(url).build();
    response = HttpRequests.execute(request);
    Assert.assertEquals(500, response.getResponseCode());
    Assert.assertTrue(response.getResponseBodyAsString().contains("Exception"));
    url = new URL(serviceURL, "verifyClassLoader");
    request = HttpRequest.get(url).build();
    response = HttpRequests.execute(request);
    Assert.assertEquals(200, response.getResponseCode());
    RuntimeMetrics serviceMetrics = serviceManager.getMetrics();
    serviceMetrics.waitForinput(3, 5, TimeUnit.SECONDS);
    Assert.assertEquals(3, serviceMetrics.getInput());
    Assert.assertEquals(2, serviceMetrics.getProcessed());
    Assert.assertEquals(1, serviceMetrics.getException());
    RuntimeMetrics handlerMetrics = getMetricsManager().getServiceHandlerMetrics(Id.Namespace.DEFAULT.getId(),
    AppWithServices.APP_NAME,
    AppWithServices.SERVICE_NAME,
    AppWithServices.SERVICE_NAME);
    handlerMetrics.waitForinput(3, 5, TimeUnit.SECONDS);
    Assert.assertEquals(3, handlerMetrics.getInput());
    Assert.assertEquals(2, handlerMetrics.getProcessed());
    Assert.assertEquals(1, handlerMetrics.getException());
    LOG.info("DatasetUpdateService Started");
    Map<String, String> args
    = ImmutableMap.of(AppWithServices.WRITE_VALUE_RUN_KEY, AppWithServices.DATASET_TEST_VALUE,
    AppWithServices.WRITE_VALUE_STOP_KEY, AppWithServices.DATASET_TEST_VALUE_STOP);
    ServiceManager datasetWorkerServiceManager = applicationManager
    .getServiceManager(AppWithServices.DATASET_WORKER_SERVICE_NAME).start(args);
    WorkerManager datasetWorker =
    applicationManager.getWorkerManager(AppWithServices.DATASET_UPDATE_WORKER).start(args);
    datasetWorkerServiceManager.waitForStatus(true);
    ServiceManager noopManager = applicationManager.getServiceManager("NoOpService").start();
    serviceManager.waitForStatus(true, 2, 1);
    String result = callServiceGet(noopManager.getServiceURL(), "ping/" + AppWithServices.DATASET_TEST_KEY);
    String decodedResult = new Gson().fromJson(result, String.class);
    Assert.assertEquals(AppWithServices.DATASET_TEST_VALUE, decodedResult);
    handlerMetrics = getMetricsManager().getServiceHandlerMetrics(Id.Namespace.DEFAULT.getId(),
    AppWithServices.APP_NAME,
    "NoOpService",
    "NoOpHandler");
    handlerMetrics.waitForinput(1, 5, TimeUnit.SECONDS);
    Assert.assertEquals(1, handlerMetrics.getInput());
    Assert.assertEquals(1, handlerMetrics.getProcessed());
    Assert.assertEquals(0, handlerMetrics.getException());
    String path = String.format("discover/%s/%s",
    AppWithServices.APP_NAME, AppWithServices.DATASET_WORKER_SERVICE_NAME);
    url = new URL(serviceURL, path);
    request = HttpRequest.get(url).build();
    response = HttpRequests.execute(request);
    Assert.assertEquals(200, response.getResponseCode());
    datasetWorker.stop();
    datasetWorkerServiceManager.stop();
    datasetWorkerServiceManager.waitForStatus(false);
    LOG.info("DatasetUpdateService Stopped");
    serviceManager.stop();
    serviceManager.waitForStatus(false);
    LOG.info("ServerService Stopped");
    result = callServiceGet(noopManager.getServiceURL(), "ping/" + AppWithServices.DATASET_TEST_KEY_STOP);
    decodedResult = new Gson().fromJson(result, String.class);
    Assert.assertEquals(AppWithServices.DATASET_TEST_VALUE_STOP, decodedResult);
    result = callServiceGet(noopManager.getServiceURL(), "ping/" + AppWithServices.DATASET_TEST_KEY_STOP_2);
    decodedResult = new Gson().fromJson(result, String.class);
    Assert.assertEquals(AppWithServices.DATASET_TEST_VALUE_STOP_2, decodedResult);
}
```

## uid 921e01caf499  (id 364, P_UNORD=1&flaky)

- project: `abel533_Mapper`
- test: `IdTest.testCompositeKeys`
- label: `unordered collections`   parse_ok: 1
- analyser: {'P_ASYNC': 0, 'P_UNORD': 1, 'P_CLOCK': 0}
- witness: {'P_UNORD': 'Assert.assertTrue(column.isId())'}

```java
@Test
public void testCompositeKeys() {
    EntityHelper.initEntityNameMap(UserCompositeKeys.class, config);
    EntityTable entityTable = EntityHelper.getEntityTable(UserCompositeKeys.class);
    Assert.assertNotNull(entityTable);
    Set<EntityColumn> columns = entityTable.getEntityClassColumns();
    Assert.assertEquals(2, columns.size());
    Assert.assertEquals(2, entityTable.getEntityClassPKColumns().size());
    for (EntityColumn column : columns) {
        Assert.assertTrue(column.isId());
    }
    ResultMap resultMap = entityTable.getResultMap(configuration);
    Assert.assertEquals(2, resultMap.getResultMappings().size());
    Assert.assertTrue(resultMap.getResultMappings().get(0).getFlags().contains(ID));
    Assert.assertTrue(resultMap.getResultMappings().get(1).getFlags().contains(ID));
    Assert.assertEquals("<where> AND name = #{name} AND orgId = #{orgId}</where>", SqlHelper.wherePKColumns(UserCompositeKeys.class));
}
```

## uid 45b7b069571d  (id 26239, topup)

- project: `Ericsson_ecchronos`
- test: `TestTableRepairJob.testPostExecuteRepairedWithFailure`
- label: `non-flaky`   parse_ok: 1
- analyser: {'P_ASYNC': 0, 'P_UNORD': 0, 'P_CLOCK': 0}
- witness: {}

```java
@Test
public void testPostExecuteRepairedWithFailure()
{
    // mock
    long repairedAt = System.currentTimeMillis();
    doReturn(repairedAt).when(myRepairStateSnapshot).lastCompletedAt();
    doReturn(false).when(myRepairStateSnapshot).canRepair();

    myRepairJob.postExecute(false, null);

    assertThat(myRepairJob.getLastSuccessfulRun()).isEqualTo(repairedAt);
    verify(myRepairState, times(1)).update();
}
```

## uid 0309b2314412  (id 26243, none&non_flaky)

- project: `Ericsson_ecchronos`
- test: `TestTableRepairJob.testGetView`
- label: `non-flaky`   parse_ok: 1
- analyser: {'P_ASYNC': 0, 'P_UNORD': 0, 'P_CLOCK': 0}
- witness: {}

```java
@Test
public void testGetView()
{
    VnodeRepairState vnodeRepairState = TestUtils.createVnodeRepairState(1, 2, ImmutableSet.of(), System.currentTimeMillis());
    VnodeRepairStatesImpl vnodeRepairStates = VnodeRepairStatesImpl.newBuilder(Arrays.asList(vnodeRepairState)).build();
    when(myRepairStateSnapshot.getVnodeRepairStates()).thenReturn(vnodeRepairStates);
    RepairJobView repairJobView = myRepairJob.getView();

    assertThat(repairJobView.getId()).isEqualTo(myTableReference.getId());
    assertThat(repairJobView.getTableReference()).isEqualTo(myTableReference);
    assertThat(repairJobView.getRepairConfiguration()).isEqualTo(myRepairConfiguration);
    assertThat(repairJobView.getRepairStateSnapshot()).isEqualTo(myRepairStateSnapshot);
    assertThat(repairJobView.getStatus()).isEqualTo(RepairJobView.Status.ERROR);
}
```

## uid 3d76f898d074  (id 35660, P_ASYNC=1&non_flaky)

- project: `cdapio_cdap`
- test: `Spark2Test.testSpark2Service`
- label: `non-flaky`   parse_ok: 1
- analyser: {'P_ASYNC': 1, 'P_UNORD': 0, 'P_CLOCK': 0}
- witness: {'P_ASYNC': 'Assert.assertNotNull(url)'}

```java
@Test
public void testSpark2Service() throws Exception {
  ApplicationManager applicationManager = deploy(NamespaceId.DEFAULT, Spark2TestApp.class);
  SparkManager manager = applicationManager.getSparkManager(ScalaSparkServiceProgram.class.getSimpleName()).start();

  URL url = manager.getServiceURL(5, TimeUnit.MINUTES);
  Assert.assertNotNull(url);

  // GET request to sum n numbers.
  URL sumURL = url.toURI().resolve("sum?n=" + Joiner.on("&n=").join(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)).toURL();
  HttpURLConnection urlConn = (HttpURLConnection) sumURL.openConnection();
  Assert.assertEquals(HttpURLConnection.HTTP_OK, urlConn.getResponseCode());
  try (InputStream is = urlConn.getInputStream()) {
    Assert.assertEquals(55, Integer.parseInt(new String(ByteStreams.toByteArray(is), StandardCharsets.UTF_8)));
  }
}
```

## uid 0c85ea8971af  (id 35698, P_CLOCK=1&non_flaky)

- project: `cdapio_cdap`
- test: `LoggingEventSerializerTest.testDecodeTimestamp`
- label: `non-flaky`   parse_ok: 1
- analyser: {'P_ASYNC': 0, 'P_UNORD': 0, 'P_CLOCK': 1}
- witness: {'P_CLOCK': 'Assert.assertEquals(timestamp, serializer.decodeEventTimestamp(ByteBuffer.wrap(bytes)))'}

```java
@Test
public void testDecodeTimestamp() throws IOException {
  long timestamp = System.currentTimeMillis();

  ch.qos.logback.classic.spi.LoggingEvent event = new ch.qos.logback.classic.spi.LoggingEvent();
  event.setLevel(Level.INFO);
  event.setLoggerName("test.logger");
  event.setMessage("Some test");
  event.setTimeStamp(timestamp);

  // Serialize it
  LoggingEventSerializer serializer = new LoggingEventSerializer();
  byte[] bytes = serializer.toBytes(event);

  // Decode timestamp
  Assert.assertEquals(timestamp, serializer.decodeEventTimestamp(ByteBuffer.wrap(bytes)));
}
```

## uid 6fac0fe57e56  (id 35708, P_UNORD=1&non_flaky)

- project: `cdapio_cdap`
- test: `CDAPLogAppenderTest.testCDAPLogAppender`
- label: `non-flaky`   parse_ok: 1
- analyser: {'P_ASYNC': 1, 'P_UNORD': 1, 'P_CLOCK': 0}
- witness: {'P_ASYNC': 'Assert.assertEquals(1, files.size())', 'P_UNORD': 'Assert.assertEquals(expectedPermissions, location.getPermissions())'}

```java
@Test
public void testCDAPLogAppender() {
  int syncInterval = 1024 * 1024;
  CDAPLogAppender cdapLogAppender = new CDAPLogAppender();

  cdapLogAppender.setSyncIntervalBytes(syncInterval);
  cdapLogAppender.setMaxFileLifetimeMs(TimeUnit.DAYS.toMillis(1));
  cdapLogAppender.setMaxFileSizeInBytes(104857600);
  cdapLogAppender.setDirPermissions("700");
  cdapLogAppender.setFilePermissions("600");
  cdapLogAppender.setFileRetentionDurationDays(1);
  cdapLogAppender.setLogCleanupIntervalMins(10);
  cdapLogAppender.setFileCleanupBatchSize(100);
  AppenderContext context = new LocalAppenderContext(injector.getInstance(TransactionRunner.class),
                                                     injector.getInstance(LocationFactory.class),
                                                     new NoOpMetricsCollectionService());
  context.start();
  cdapLogAppender.setContext(context);
  cdapLogAppender.start();

  FileMetaDataReader fileMetaDataReader = injector.getInstance(FileMetaDataReader.class);
  LoggingEvent event =
    new LoggingEvent("io.cdap.Test",
                     (ch.qos.logback.classic.Logger) LoggerFactory.getLogger(Logger.ROOT_LOGGER_NAME),
                     Level.ERROR , "test message", null, null);
  Map<String, String> properties = new HashMap<>();
  properties.put(NamespaceLoggingContext.TAG_NAMESPACE_ID, "default");
  properties.put(ApplicationLoggingContext.TAG_APPLICATION_ID, "testApp");
  properties.put(UserServiceLoggingContext.TAG_USER_SERVICE_ID, "testService");

  event.setMDCPropertyMap(properties);

  cdapLogAppender.doAppend(event);
  cdapLogAppender.stop();
  context.stop();

  try {
    List<LogLocation> files = fileMetaDataReader.listFiles(cdapLogAppender.getLoggingPath(properties),
                                                           0, Long.MAX_VALUE);
    Assert.assertEquals(1, files.size());
    LogLocation logLocation = files.get(0);
    Assert.assertEquals(LogLocation.VERSION_1, logLocation.getFrameworkVersion());
    Assert.assertTrue(logLocation.getLocation().exists());
    CloseableIterator<LogEvent> logEventCloseableIterator =
      logLocation.readLog(Filter.EMPTY_FILTER, 0, Long.MAX_VALUE, Integer.MAX_VALUE);
    int logCount = 0;
    while (logEventCloseableIterator.hasNext()) {
      logCount++;
      LogEvent logEvent = logEventCloseableIterator.next();
      Assert.assertEquals(event.getMessage(), logEvent.getLoggingEvent().getMessage());
    }
    logEventCloseableIterator.close();
    Assert.assertEquals(1, logCount);
    // checking permission
    String expectedPermissions = "rw-------";
    for (LogLocation file : files) {
      Location location = file.getLocation();
      Assert.assertEquals(expectedPermissions, location.getPermissions());
    }
  } catch (Exception e) {
    Assert.fail();
  }
}
```

## uid e97e8586dfba  (id 38234, P_UNORD=1&non_flaky)

- project: `palantir_atlasdb`
- test: `AbstractAtlasDbKeyValueServiceTest.testGetRowColumnSelection`
- label: `non-flaky`   parse_ok: 1
- analyser: {'P_ASYNC': 0, 'P_UNORD': 1, 'P_CLOCK': 0}
- witness: {'P_UNORD': 'Assert.assertEquals(ImmutableSet.of(cell1, cell2, cell3), rows1.keySet())'}

```java
@Test
public void testGetRowColumnSelection() {
    Cell cell1 = Cell.create(PtBytes.toBytes("row"), PtBytes.toBytes("col1"));
    Cell cell2 = Cell.create(PtBytes.toBytes("row"), PtBytes.toBytes("col2"));
    Cell cell3 = Cell.create(PtBytes.toBytes("row"), PtBytes.toBytes("col3"));
    byte[] val = PtBytes.toBytes("val");

    keyValueService.put(TEST_TABLE, ImmutableMap.of(cell1, val, cell2, val, cell3, val), 0);

    Map<Cell, Value> rows1 = keyValueService.getRows(
            TEST_TABLE,
            ImmutableSet.of(cell1.getRowName()),
            ColumnSelection.all(),
            1);
    Assert.assertEquals(ImmutableSet.of(cell1, cell2, cell3), rows1.keySet());

    Map<Cell, Value> rows2 = keyValueService.getRows(
            TEST_TABLE,
            ImmutableSet.of(cell1.getRowName()),
            ColumnSelection.create(ImmutableList.of(cell1.getColumnName())),
            1);
    assertEquals(ImmutableSet.of(cell1), rows2.keySet());

    Map<Cell, Value> rows3 = keyValueService.getRows(
            TEST_TABLE,
            ImmutableSet.of(cell1.getRowName()),
            ColumnSelection.create(ImmutableList.of(cell1.getColumnName(), cell3.getColumnName())),
            1);
    assertEquals(ImmutableSet.of(cell1, cell3), rows3.keySet());
    Map<Cell, Value> rows4 = keyValueService.getRows(
            TEST_TABLE,
            ImmutableSet.of(cell1.getRowName()),
            ColumnSelection.create(ImmutableList.<byte[]>of()),
            1);

    // This has changed recently - now empty column set means
    // that all columns are selected.
    assertEquals(ImmutableSet.of(cell1, cell2, cell3), rows4.keySet());
}
```

## uid b3022c55336a  (id 43122, P_ASYNC=1&non_flaky)

- project: `trinodb_trino`
- test: `BaseConnectorSmokeTest.testDeleteAllDataFromTable`
- label: `non-flaky`   parse_ok: 1
- analyser: {'P_ASYNC': 1, 'P_UNORD': 0, 'P_CLOCK': 0}
- witness: {'P_ASYNC': 'assertQuery("SELECT count(*) FROM " + table.getName(), "VALUES 0")'}

```java
@Test
public void testDeleteAllDataFromTable()
{
    skipTestUnless(hasBehavior(SUPPORTS_CREATE_TABLE) && hasBehavior(SUPPORTS_DELETE));
    try (TestTable table = new TestTable(getQueryRunner()::execute, "test_delete_all_data", "AS SELECT * FROM region")) {
        // not using assertUpdate as some connectors provide update count and some do not
        getQueryRunner().execute("DELETE FROM " + table.getName());
        assertQuery("SELECT count(*) FROM " + table.getName(), "VALUES 0");
    }
}
```

## uid dfc89f30cbee  (id 59639, topup)

- project: `looly_hutool`
- test: `TokenizerUtilTest.smartcnTest`
- label: `non-flaky`   parse_ok: 1
- analyser: {'P_ASYNC': 0, 'P_UNORD': 0, 'P_CLOCK': 0}
- witness: {}

```java
@Test
public void smartcnTest() {
    TokenizerEngine engine = new SmartcnEngine();
    Result result = engine.parse(text);
    String resultStr = IterUtil.join((Iterator<Word>)result, " ");
    Assert.assertEquals("è¿ ä¸¤ ä¸ª æ¹æ³ ç åºå« å¨äº è¿å å¼", resultStr);
}
```

## uid 9eba8d9d8f61  (id 70770, P_ASYNC=1&non_flaky)

- project: `apache_kafka`
- test: `ConnectWorkerIntegrationTest.testRestartFailedTask`
- label: `non-flaky`   parse_ok: 1
- analyser: {'P_ASYNC': 1, 'P_UNORD': 0, 'P_CLOCK': 0}
- witness: {'P_ASYNC': 'assertWorkersUp(NUM_WORKERS)'}

```java
@Test
public void testRestartFailedTask() throws Exception {
    connect = connectBuilder.build();
    // start the clusters
    connect.start();

    int numTasks = 1;

    // Properties for the source connector. The task should fail at startup due to the bad broker address.
    Map<String, String> connectorProps = new HashMap<>();
    connectorProps.put(CONNECTOR_CLASS_CONFIG, MonitorableSourceConnector.class.getName());
    connectorProps.put(TASKS_MAX_CONFIG, Objects.toString(numTasks));
    connectorProps.put(CONNECTOR_CLIENT_PRODUCER_OVERRIDES_PREFIX + BOOTSTRAP_SERVERS_CONFIG, "nobrokerrunningatthisaddress");

    waitForCondition(() -> assertWorkersUp(NUM_WORKERS).orElse(false),
            WORKER_SETUP_DURATION_MS, "Initial group of workers did not start in time.");

    // Try to start the connector and its single task.
    connect.configureConnector(CONNECTOR_NAME, connectorProps);

    waitForCondition(() -> assertConnectorTasksFailed(CONNECTOR_NAME, numTasks).orElse(false),
            CONNECTOR_SETUP_DURATION_MS, "Connector tasks did not fail in time");

    // Reconfigure the connector without the bad broker address.
    connectorProps.remove(CONNECTOR_CLIENT_PRODUCER_OVERRIDES_PREFIX + BOOTSTRAP_SERVERS_CONFIG);
    connect.configureConnector(CONNECTOR_NAME, connectorProps);

    // Restart the failed task
    String taskRestartEndpoint = connect.endpointForResource(
        String.format("connectors/%s/tasks/0/restart", CONNECTOR_NAME));
    connect.executePost(taskRestartEndpoint, "", Collections.emptyMap());

    // Ensure the task started successfully this time
    waitForCondition(() -> assertConnectorAndTasksRunning(CONNECTOR_NAME, numTasks).orElse(false),
        CONNECTOR_SETUP_DURATION_MS, "Connector tasks are not all in running state.");
}
```

## uid 5550ff0f1a58  (id 70771, P_ASYNC=1&non_flaky)

- project: `apache_kafka`
- test: `ConnectWorkerIntegrationTest.testBrokerCoordinator`
- label: `non-flaky`   parse_ok: 1
- analyser: {'P_ASYNC': 1, 'P_UNORD': 0, 'P_CLOCK': 0}
- witness: {'P_ASYNC': 'assertWorkersUp(NUM_WORKERS)'}

```java
@Test
public void testBrokerCoordinator() throws Exception {
    workerProps.put(DistributedConfig.SCHEDULED_REBALANCE_MAX_DELAY_MS_CONFIG, String.valueOf(5000));
    connect = connectBuilder.workerProps(workerProps).build();
    // start the clusters
    connect.start();
    int numTasks = 4;
    // create test topic
    connect.kafka().createTopic("test-topic", NUM_TOPIC_PARTITIONS);

    // setup up props for the sink connector
    Map<String, String> props = new HashMap<>();
    props.put(CONNECTOR_CLASS_CONFIG, MonitorableSourceConnector.class.getSimpleName());
    props.put(TASKS_MAX_CONFIG, String.valueOf(numTasks));
    props.put("topic", "test-topic");
    props.put("throughput", String.valueOf(1));
    props.put("messages.per.poll", String.valueOf(10));
    props.put(KEY_CONVERTER_CLASS_CONFIG, StringConverter.class.getName());
    props.put(VALUE_CONVERTER_CLASS_CONFIG, StringConverter.class.getName());

    waitForCondition(() -> assertWorkersUp(NUM_WORKERS).orElse(false),
            WORKER_SETUP_DURATION_MS, "Initial group of workers did not start in time.");

    // start a source connector
    connect.configureConnector(CONNECTOR_NAME, props);

    waitForCondition(() -> assertConnectorAndTasksRunning(CONNECTOR_NAME, numTasks).orElse(false),
            CONNECTOR_SETUP_DURATION_MS, "Connector tasks did not start in time.");

    connect.kafka().stopOnlyKafka();

    waitForCondition(() -> assertWorkersUp(NUM_WORKERS).orElse(false),
            WORKER_SETUP_DURATION_MS, "Group of workers did not remain the same after broker shutdown");

    // Allow for the workers to discover that the coordinator is unavailable, wait is
    // heartbeat timeout * 2 + 4sec
    Thread.sleep(TimeUnit.SECONDS.toMillis(10));

    connect.kafka().startOnlyKafkaOnSamePorts();

    // Allow for the kafka brokers to come back online
    Thread.sleep(TimeUnit.SECONDS.toMillis(10));

    waitForCondition(() -> assertWorkersUp(NUM_WORKERS).orElse(false),
            WORKER_SETUP_DURATION_MS, "Group of workers did not remain the same within the "
                    + "designated time.");

    // Allow for the workers to rebalance and reach a steady state
    Thread.sleep(TimeUnit.SECONDS.toMillis(10));

    waitForCondition(() -> assertConnectorAndTasksRunning(CONNECTOR_NAME, numTasks).orElse(false),
            CONNECTOR_SETUP_DURATION_MS, "Connector tasks did not start in time.");
}
```

## uid 95d94d5fc3ea  (id 76969, none&non_flaky)

- project: `Tencent_Firestorm`
- test: `ArgumentsTest.argTest`
- label: `non-flaky`   parse_ok: 1
- analyser: {'P_ASYNC': 0, 'P_UNORD': 0, 'P_CLOCK': 0}
- witness: {}

```java
@Test
public void argTest() {
  String[] args = {"-c", confFile};
  Arguments arguments = new Arguments();
  CommandLine commandLine = new CommandLine(arguments);
  commandLine.parseArgs(args);
  assertEquals(confFile, arguments.getConfigFile());
}
```

## uid fdbd1aa2cf96  (id 78236, none&non_flaky)

- project: `apache_beam`
- test: `SimplePushbackSideInputDoFnRunnerTest.processElementSideInputReadyAllWindows`
- label: `non-flaky`   parse_ok: 1
- analyser: {'P_ASYNC': 0, 'P_UNORD': 0, 'P_CLOCK': 0}
- witness: {}

```java
@Test
public void processElementSideInputReadyAllWindows() {
  when(reader.isReady(Mockito.eq(singletonView), Mockito.any(BoundedWindow.class)))
      .thenReturn(true);

  ImmutableList<PCollectionView<?>> views = ImmutableList.of(singletonView);
  SimplePushbackSideInputDoFnRunner<Integer, Integer> runner = createRunner(views);

  WindowedValue<Integer> multiWindow =
      WindowedValue.of(
          2,
          new Instant(-2),
          ImmutableList.of(
              new IntervalWindow(new Instant(-500L), new Instant(0L)),
              new IntervalWindow(BoundedWindow.TIMESTAMP_MIN_VALUE, new Instant(250L)),
              GlobalWindow.INSTANCE),
          PaneInfo.ON_TIME_AND_ONLY_FIRING);
  Iterable<WindowedValue<Integer>> multiWindowPushback =
      runner.processElementInReadyWindows(multiWindow);
  assertThat(multiWindowPushback, emptyIterable());
  assertThat(
      underlying.inputElems,
      containsInAnyOrder(ImmutableList.copyOf(multiWindow.explodeWindows()).toArray()));
}
```

## uid dda60271f595  (id 86059, P_ASYNC=1&non_flaky)

- project: `graylog2_graylog2-server`
- test: `AggregationEventProcessorConfigTest.toJobSchedulerConfig`
- label: `non-flaky`   parse_ok: 1
- analyser: {'P_ASYNC': 1, 'P_UNORD': 0, 'P_CLOCK': 0}
- witness: {'P_ASYNC': 'assertThat(schedulerConfig.schedule())'}

```java
@Test
public void toJobSchedulerConfig() {
    final EventDefinitionDto dto = dbService.get("54e3deadbeefdeadbeefaffe").orElse(null);

    assertThat(dto).isNotNull();

    assertThat(dto.config().toJobSchedulerConfig(dto, clock)).isPresent().get().satisfies(schedulerConfig -> {
        assertThat(schedulerConfig.jobDefinitionConfig()).satisfies(jobDefinitionConfig -> {
            assertThat(jobDefinitionConfig).isInstanceOf(EventProcessorExecutionJob.Config.class);

            final EventProcessorExecutionJob.Config config = (EventProcessorExecutionJob.Config) jobDefinitionConfig;

            assertThat(config.eventDefinitionId()).isEqualTo(dto.id());
            assertThat(config.processingWindowSize()).isEqualTo(300000);
            assertThat(config.processingHopSize()).isEqualTo(300000);
            assertThat(config.parameters()).isEqualTo(AggregationEventProcessorParameters.builder()
                    .timerange(AbsoluteRange.create(clock.nowUTC().minus(300000), clock.nowUTC()))
                    .build());
        });

        assertThat(schedulerConfig.schedule()).satisfies(schedule -> {
            assertThat(schedule).isInstanceOf(IntervalJobSchedule.class);

            final IntervalJobSchedule config = (IntervalJobSchedule) schedule;

            assertThat(config.interval()).isEqualTo(300000);
            assertThat(config.unit()).isEqualTo(TimeUnit.MILLISECONDS);
        });
    });
}
```

## uid eb8358f80bad  (id 88822, P_CLOCK=1&non_flaky)

- project: `apache_ignite`
- test: `IgniteThrottlingUnitTest.averageCalculation`
- label: `non-flaky`   parse_ok: 1
- analyser: {'P_ASYNC': 0, 'P_UNORD': 0, 'P_CLOCK': 1}
- witness: {'P_CLOCK': 'assertEquals(0, measurement.getSpeedOpsPerSec(System.nanoTime()))'}

```java
@Test
public void averageCalculation() throws InterruptedException {
    IntervalBasedMeasurement measurement = new IntervalBasedMeasurement(100, 1);

    for (int i = 0; i < 1000; i++)
        measurement.addMeasurementForAverageCalculation(100);

    assertEquals(100, measurement.getAverage());

    Thread.sleep(220);

    assertEquals(0, measurement.getAverage());

    assertEquals(0, measurement.getSpeedOpsPerSec(System.nanoTime()));
}
```

## uid 2ad3a4dc0234  (id 89274, P_ASYNC=1&non_flaky)

- project: `apache_samza`
- test: `TestSamzaRestService.testStartShouldStartTheMetricsReportersAndServer`
- label: `non-flaky`   parse_ok: 1
- analyser: {'P_ASYNC': 1, 'P_UNORD': 0, 'P_CLOCK': 0}
- witness: {'P_ASYNC': 'Mockito.verify(metricsReporter)'}

```java
@Test
public void testStartShouldStartTheMetricsReportersAndServer() throws Exception {
  NetworkConnector connector = Mockito.mock(NetworkConnector.class);
  int testServerPort = 100;
  Mockito.doReturn(testServerPort).when(connector).getPort();
  Mockito.when(server.getConnectors()).thenReturn(new NetworkConnector[]{connector});
  Mockito.doNothing().when(server).start();
  samzaRestService.start();
  Mockito.verify(metricsReporter).start();
  Mockito.verify(metricsReporter).register("SamzaRest", metricsRegistry);
  Mockito.verify(server).start();
}
```

## uid 7cdbe9c47610  (id 97709, none&non_flaky)

- project: `vojtechhabarta_typescript-generator`
- test: `CustomTypeAliasesTest.testNonGeneric`
- label: `non-flaky`   parse_ok: 1
- analyser: {'P_ASYNC': 0, 'P_UNORD': 0, 'P_CLOCK': 0}
- witness: {}

```java
@Test
public void testNonGeneric() {
    final Settings settings = TestUtils.settings();
    settings.customTypeAliases = Collections.singletonMap("Id", "string");
    final String output = new TypeScriptGenerator(settings).generateTypeScript(Input.from());
    Assert.assertTrue(output.contains("type Id = string"));
}
```

## uid dcf6809df62b  (id 98317, P_UNORD=1&non_flaky)

- project: `spotify_docker-client`
- test: `DefaultDockerClientUnitTest.testMultipleHeaders`
- label: `non-flaky`   parse_ok: 1
- analyser: {'P_ASYNC': 0, 'P_UNORD': 1, 'P_CLOCK': 0}
- witness: {'P_UNORD': 'Assert.assertEquals(entry.getKey(), nameCaptor.getAllValues().get(idx))'}

```java
@Test
public void testMultipleHeaders() throws Exception {
  final Map<String, Object> headers = Maps.newHashMap();
  headers.put("int", 1);
  headers.put("string", "2");
  headers.put("list", Lists.newArrayList("a", "b", "c"));

  for (final Map.Entry<String, Object> entry : headers.entrySet()) {
    builder.header(entry.getKey(), entry.getValue());
  }

  final DefaultDockerClient dockerClient = new DefaultDockerClient(
      builder, clientBuilderSupplier);
  dockerClient.info();

  final ArgumentCaptor<String> nameCaptor = ArgumentCaptor.forClass(String.class);
  final ArgumentCaptor<String> valueCaptor = ArgumentCaptor.forClass(String.class);
  verify(builderMock, times(headers.size())).header(nameCaptor.capture(), valueCaptor.capture());

  int idx = 0;
  for (final Map.Entry<String, Object> entry : headers.entrySet()) {
    Assert.assertEquals(entry.getKey(), nameCaptor.getAllValues().get(idx));
    Assert.assertEquals(entry.getValue(), valueCaptor.getAllValues().get(idx));
    ++idx;
  }
}
```

## uid 7413a819527f  (id 104170, none&non_flaky)

- project: `spring-cloud_spring-cloud-config`
- test: `EnvironmentPrefixHelperTests.testKeysDefaults`
- label: `non-flaky`   parse_ok: 1
- analyser: {'P_ASYNC': 0, 'P_UNORD': 0, 'P_CLOCK': 0}
- witness: {}

```java
@Test
public void testKeysDefaults() {
    Map<String, String> keys = this.helper.getEncryptorKeys("foo", "bar", "spam");
    assertThat(keys.get("name")).isEqualTo("foo");
    assertThat(keys.get("profiles")).isEqualTo("bar");
}
```

## uid 342bbe255aa8  (id 104660, topup)

- project: `apache_pinot`
- test: `OfflineClusterIntegrationTest.testQuerySourceWithDatabaseName`
- label: `non-flaky`   parse_ok: 1
- analyser: {'P_ASYNC': 0, 'P_UNORD': 0, 'P_CLOCK': 0}
- witness: {}

```java
@Test
public void testQuerySourceWithDatabaseName()
    throws Exception {
  // by default 10 rows will be returned, so use high limit
  String pql = "SELECT DISTINCT(Carrier) FROM mytable LIMIT 1000000";
  String sql = "SELECT DISTINCT Carrier FROM mytable";
  testQuery(pql, Collections.singletonList(sql));
  pql = "SELECT DISTINCT Carrier FROM db.mytable LIMIT 1000000";
  testSqlQuery(pql, Collections.singletonList(sql));
}
```

## uid 9dd7707f209b  (id 104691, P_UNORD=1&non_flaky)

- project: `apache_pinot`
- test: `ConvertToRawIndexMinionClusterIntegrationTest.testConvertToRawIndexTask`
- label: `non-flaky`   parse_ok: 1
- analyser: {'P_ASYNC': 0, 'P_UNORD': 1, 'P_CLOCK': 0}
- witness: {'P_UNORD': 'Assert.assertNotNull(indexDirs)'}

```java
@Test
public void testConvertToRawIndexTask()
    throws Exception {
  String offlineTableName = TableNameBuilder.OFFLINE.tableNameWithType(getTableName());

  File testDataDir = new File(CommonConstants.Server.DEFAULT_INSTANCE_DATA_DIR + "-0", offlineTableName);
  if (!testDataDir.isDirectory()) {
    testDataDir = new File(CommonConstants.Server.DEFAULT_INSTANCE_DATA_DIR + "-1", offlineTableName);
  }
  Assert.assertTrue(testDataDir.isDirectory());
  File tableDataDir = testDataDir;

  // Check that all columns have dictionary
  File[] indexDirs = tableDataDir.listFiles();
  Assert.assertNotNull(indexDirs);
  for (File indexDir : indexDirs) {
    SegmentMetadata segmentMetadata = new SegmentMetadataImpl(indexDir);
    for (String columnName : segmentMetadata.getSchema().getColumnNames()) {
      Assert.assertTrue(segmentMetadata.getColumnMetadataFor(columnName).hasDictionary());
    }
  }

  // Should create the task queues and generate a ConvertToRawIndexTask task with 5 child tasks
  Assert.assertNotNull(_taskManager.scheduleTasks().get(ConvertToRawIndexTask.TASK_TYPE));
  Assert.assertTrue(_helixTaskResourceManager.getTaskQueues()
      .contains(PinotHelixTaskResourceManager.getHelixJobQueueName(ConvertToRawIndexTask.TASK_TYPE)));

  // Should generate one more ConvertToRawIndexTask task with 3 child tasks
  Assert.assertNotNull(_taskManager.scheduleTasks().get(ConvertToRawIndexTask.TASK_TYPE));

  // Should not generate more tasks
  Assert.assertNull(_taskManager.scheduleTasks().get(ConvertToRawIndexTask.TASK_TYPE));

  // Wait at most 600 seconds for all tasks COMPLETED and new segments refreshed
  TestUtils.waitForCondition(input -> {
    // Check task state
    for (TaskState taskState : _helixTaskResourceManager.getTaskStates(ConvertToRawIndexTask.TASK_TYPE).values()) {
      if (taskState != TaskState.COMPLETED) {
        return false;
      }
    }

    // Check segment ZK metadata
    for (SegmentZKMetadata segmentZKMetadata : _helixResourceManager.getSegmentsZKMetadata(offlineTableName)) {
      Map<String, String> customMap = segmentZKMetadata.getCustomMap();
      if (customMap == null || customMap.size() != 1 || !customMap
          .containsKey(ConvertToRawIndexTask.TASK_TYPE + MinionConstants.TASK_TIME_SUFFIX)) {
        return false;
      }
    }

    // Check segment metadata
    File[] indexDirs1 = tableDataDir.listFiles();
    Assert.assertNotNull(indexDirs1);
    for (File indexDir : indexDirs1) {
      SegmentMetadata segmentMetadata;

      // Segment metadata file might not exist if the segment is refreshing
      try {
        segmentMetadata = new SegmentMetadataImpl(indexDir);
      } catch (Exception e) {
        return false;
      }

      // The columns in COLUMNS_TO_CONVERT should have raw index
      List<String> rawIndexColumns = Arrays.asList(StringUtils.split(COLUMNS_TO_CONVERT, ','));
      for (String columnName : segmentMetadata.getSchema().getColumnNames()) {
        if (rawIndexColumns.contains(columnName)) {
          if (segmentMetadata.getColumnMetadataFor(columnName).hasDictionary()) {
            return false;
          }
        } else {
          if (!segmentMetadata.getColumnMetadataFor(columnName).hasDictionary()) {
            return false;
          }
        }
      }
    }

    return true;
  }, 600_000L, "Failed to get all tasks COMPLETED and new segments refreshed");
}
```

## uid 50380f942455  (id 112115, P_CLOCK=1&non_flaky)

- project: `apache_shardingsphere-elasticjob`
- test: `StatisticRdbRepositoryTest.assertGetSummedTaskResultStatisticsWhenTableIsEmpty`
- label: `non-flaky`   parse_ok: 1
- analyser: {'P_ASYNC': 0, 'P_UNORD': 0, 'P_CLOCK': 1}
- witness: {'P_CLOCK': 'assertThat(po.getSuccessCount(), is(0))'}

```java
@Test
public void assertGetSummedTaskResultStatisticsWhenTableIsEmpty() {
    for (StatisticInterval each : StatisticInterval.values()) {
        TaskResultStatistics po = repository.getSummedTaskResultStatistics(new Date(), each);
        assertThat(po.getSuccessCount(), is(0));
        assertThat(po.getFailedCount(), is(0));
    }
}
```

## uid 8eeba58b288d  (id 112120, P_CLOCK=1&non_flaky)

- project: `apache_shardingsphere-elasticjob`
- test: `StatisticRdbRepositoryTest.assertFindTaskRunningStatisticsWithDifferentFromDate`
- label: `non-flaky`   parse_ok: 1
- analyser: {'P_ASYNC': 0, 'P_UNORD': 0, 'P_CLOCK': 1}
- witness: {'P_CLOCK': 'assertTrue(repository.add(new TaskRunningStatistics(100, now)))'}

```java
@Test
public void assertFindTaskRunningStatisticsWithDifferentFromDate() {
    Date now = new Date();
    Date yesterday = getYesterday();
    assertTrue(repository.add(new TaskRunningStatistics(100, yesterday)));
    assertTrue(repository.add(new TaskRunningStatistics(100, now)));
    assertThat(repository.findTaskRunningStatistics(yesterday).size(), is(2));
    assertThat(repository.findTaskRunningStatistics(now).size(), is(1));
}
```

## uid 9b3d73f044c4  (id 112156, P_CLOCK=1&non_flaky)

- project: `apache_shardingsphere-elasticjob`
- test: `TimeServiceTest.assertGetCurrentMillis`
- label: `non-flaky`   parse_ok: 1
- analyser: {'P_ASYNC': 0, 'P_UNORD': 0, 'P_CLOCK': 1}
- witness: {'P_CLOCK': 'assertTrue(timeService.getCurrentMillis() <= System.currentTimeMillis())'}

```java
@Test
public void assertGetCurrentMillis() throws Exception {
    assertTrue(timeService.getCurrentMillis() <= System.currentTimeMillis());
}
```

## uid d551646cc40a  (id 113891, none&non_flaky)

- project: `spring-projects_spring-data-couchbase`
- test: `ReactiveCouchbaseTemplateQueryCollectionIntegrationTests.findByAnalyticsOptions`
- label: `non-flaky`   parse_ok: 1
- analyser: {'P_ASYNC': 0, 'P_UNORD': 0, 'P_CLOCK': 0}
- witness: {}

```java
@Test
public void findByAnalyticsOptions() { // 2
    AnalyticsOptions options = AnalyticsOptions.analyticsOptions().timeout(Duration.ofNanos(10));
    assertThrows(AmbiguousTimeoutException.class, () -> template.findByAnalytics(Airport.class).inScope(otherScope)
            .inCollection(otherCollection).withOptions(options).all().collectList().block());
}
```

## uid a1b7077aa17b  (id 156084, topup)

- project: `soot-oss_soot`
- test: `PropagateLineNumberTagTest.nullAssignment`
- label: `non-flaky`   parse_ok: 1
- analyser: {'P_ASYNC': 0, 'P_UNORD': 0, 'P_CLOCK': 0}
- witness: {}

```java
@Test
public void nullAssignment() {
  SootMethod target =
      prepareTarget(
          methodSigFromComponents(TEST_TARGET_CLASS, "void", "nullAssignment"),
          TEST_TARGET_CLASS);

  Body body = target.retrieveActiveBody();

  Optional<Unit> unit =
      body.getUnits().stream()
          .filter(
              u ->
                  u.toString()
                      .equals(
                          "staticinvoke <soot.jimple.PropagateLineNumberTag: soot.jimple.PropagateLineNumberTag$A foo(soot.jimple.PropagateLineNumberTag$A)>(null)"))
          .findFirst();

  assertTrue(unit.isPresent());

  List<ValueBox> useBoxes = unit.get().getUseBoxes();

  assertEquals(2, useBoxes.size());
  ValueBox valueBox = useBoxes.get(0);
  assertTrue(valueBox instanceof ImmediateBox);
  assertEquals(1, valueBox.getTags().size());
  assertTrue(valueBox.getTags().get(0) instanceof LineNumberTag);
  assertEquals(33, valueBox.getJavaSourceStartLineNumber());
}
```

## uid 2c47495afd52  (id 156414, P_CLOCK=1&non_flaky)

- project: `apache_commons-lang`
- test: `FastDateFormatTest.testLANG_1152`
- label: `non-flaky`   parse_ok: 1
- analyser: {'P_ASYNC': 0, 'P_UNORD': 0, 'P_CLOCK': 1}
- witness: {'P_CLOCK': 'assertEquals("292278994-08-17", dateAsString)'}

```java
@Test
public void testLANG_1152() {
    final TimeZone utc = FastTimeZone.getGmtTimeZone();
    final Date date = new Date(Long.MAX_VALUE);

    String dateAsString = FastDateFormat.getInstance("yyyy-MM-dd", utc, Locale.US).format(date);
    assertEquals("292278994-08-17", dateAsString);

    dateAsString = FastDateFormat.getInstance("dd/MM/yyyy", utc, Locale.US).format(date);
    assertEquals("17/08/292278994", dateAsString);
}
```

## uid 93ce1bb48ad3  (id 162604, P_UNORD=1&non_flaky)

- project: `open-telemetry_opentelemetry-java-instrumentation`
- test: `ReferenceCollectorTest.shouldCollectVirtualFields`
- label: `non-flaky`   parse_ok: 1
- analyser: {'P_ASYNC': 0, 'P_UNORD': 1, 'P_CLOCK': 0}
- witness: {'P_UNORD': 'assertThat(virtualFieldMappings.entrySet())'}

```java
@Test
public void shouldCollectVirtualFields() {
  ReferenceCollector collector = new ReferenceCollector(s -> false);
  collector.collectReferencesFromAdvice(VirtualFieldTestClasses.ValidAdvice.class.getName());
  collector.prune();

  VirtualFieldMappings virtualFieldMappings = collector.getVirtualFieldMappings();
  assertThat(virtualFieldMappings.entrySet())
      .containsExactlyInAnyOrder(
          entry(VirtualFieldTestClasses.Key1.class.getName(), Context.class.getName()),
          entry(VirtualFieldTestClasses.Key2.class.getName(), Context.class.getName()));
}
```

## uid 91cb3dc5551a  (id 175749, topup)

- project: `GoogleCloudPlatform_google-cloud-eclipse`
- test: `AppEngineDeployPreferencesPanelTest.testProjectSavedInPreferencesSelected`
- label: `non-flaky`   parse_ok: 1
- analyser: {'P_ASYNC': 0, 'P_UNORD': 0, 'P_CLOCK': 0}
- witness: {}

```java
@Test
public void testProjectSavedInPreferencesSelected()
    throws ProjectRepositoryException, InterruptedException, BackingStoreException {
  IEclipsePreferences node =
      new ProjectScope(project).getNode(DeployPreferences.PREFERENCE_STORE_QUALIFIER);
  try {
    node.put("project.id", "projectId1");
    node.put("account.email", EMAIL_1);
    initializeProjectRepository();
    when(loginService.getAccounts()).thenReturn(twoAccountSet);
    deployPanel = createPanel(true /* requireValues */);
    deployPanel.latestGcpProjectQueryJob.join();

    ProjectSelector projectSelector = getProjectSelector();
    IStructuredSelection selection = projectSelector.getViewer().getStructuredSelection();
    assertThat(selection.size(), is(1));
    assertThat(((GcpProject) selection.getFirstElement()).getId(), is("projectId1"));
  } finally {
    node.clear();
  }
}
```

## uid 773d653c4a52  (id 179428, P_UNORD=1&non_flaky)

- project: `abel533_Mapper`
- test: `ColumnTypeTest.testTypehandler`
- label: `non-flaky`   parse_ok: 1
- analyser: {'P_ASYNC': 0, 'P_UNORD': 1, 'P_CLOCK': 0}
- witness: {'P_UNORD': 'Assert.assertEquals("name", column.getColumn())'}

```java
@Test
public void testTypehandler(){
    EntityHelper.initEntityNameMap(UserTypehandler.class, config);
    EntityTable entityTable = EntityHelper.getEntityTable(UserTypehandler.class);
    Assert.assertNotNull(entityTable);

    Set<EntityColumn> columns = entityTable.getEntityClassColumns();
    Assert.assertEquals(1, columns.size());

    for (EntityColumn column : columns) {
        Assert.assertEquals("name", column.getColumn());
        Assert.assertEquals("name", column.getProperty());

        Assert.assertEquals("name = #{name, typeHandler=org.apache.ibatis.type.BlobTypeHandler}", column.getColumnEqualsHolder());
        Assert.assertEquals("name = #{record.name, typeHandler=org.apache.ibatis.type.BlobTypeHandler}", column.getColumnEqualsHolder("record"));
        Assert.assertEquals("#{name, typeHandler=org.apache.ibatis.type.BlobTypeHandler}", column.getColumnHolder());
        Assert.assertEquals("#{record.name, typeHandler=org.apache.ibatis.type.BlobTypeHandler}", column.getColumnHolder("record"));
        Assert.assertEquals("#{record.name, typeHandler=org.apache.ibatis.type.BlobTypeHandler}", column.getColumnHolder("record", "suffix"));
        Assert.assertEquals("#{record.namesuffix, typeHandler=org.apache.ibatis.type.BlobTypeHandler},", column.getColumnHolder("record", "suffix", ","));
        Assert.assertNotNull(column.getTypeHandler());
    }

    ResultMap resultMap = entityTable.getResultMap(configuration);
    Assert.assertEquals("[NAME]", resultMap.getMappedColumns().toString());

    Assert.assertEquals(1, resultMap.getResultMappings().size());

    ResultMapping resultMapping = resultMap.getResultMappings().get(0);
    Assert.assertEquals("name", resultMapping.getColumn());
    Assert.assertEquals("name", resultMapping.getProperty());
    Assert.assertNull(resultMapping.getJdbcType());
    Assert.assertEquals(BlobTypeHandler.class, resultMapping.getTypeHandler().getClass());
}
```

## uid d42fc715cb2c  (id 179477, topup)

- project: `abel533_Mapper`
- test: `SafeUpdateByFieldTest.testSafeUpdateNull`
- label: `non-flaky`   parse_ok: 1
- analyser: {'P_ASYNC': 0, 'P_UNORD': 0, 'P_CLOCK': 0}
- witness: {}

```java
@Test(expected = PersistenceException.class)
public void testSafeUpdateNull() {
    SqlSession sqlSession = getSqlSession();
    try {
        CountryMapper mapper = sqlSession.getMapper(CountryMapper.class);
        mapper.updateByExample(new Country(), null);
    } finally {
        sqlSession.close();
    }
}
```
