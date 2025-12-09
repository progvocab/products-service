# Message Delivery Semantics in Kafka

Kafka provides three delivery guarantees that determine how messages flow from Producer to Broker and from Broker to Consumer. These guarantees rely on Kafka components such as the Producer client, Broker replication layer, Consumer client, and Consumer Group Coordinator.

### At Most Once Delivery

Messages may be lost but are never re-delivered.
Producer sends data without retries or acknowledgment checks.
Consumer commits the offset before processing, so a crash loses work.
Kafka components involved: Producer client (no retry), Broker (no guarantee), Consumer client (early offset commit).

### At Least Once Delivery

Messages are never lost but may be processed more than once.
Producer retries sends until Broker acknowledges the write.
Consumer commits offset after processing, so a crash replays messages.
Kafka components involved: Producer retry logic, Broker replication layer, Consumer client (post-processing offset commit).

### Exactly Once Delivery

Messages are delivered once with no duplicates by combining idempotent producers and transactional semantics.
Producer enables idempotence and writes messages with unique Producer IDs; Broker enforces duplicate suppression.
Consumer processes transactional batches only when committed.
Kafka components involved: Idempotent Producer, Broker transaction coordinator, Consumer client with isolation.level=read_committed.

### Producer Configuration

```properties
acks=all
enable.idempotence=true
retries=5
max.in.flight.requests.per.connection=1
```

The Kafka Producer client uses idempotence to avoid duplicate writes, while the Broker transaction coordinator ensures atomic commit.

### Consumer 

```java
while (true) {
    var records = consumer.poll();
    for (var record : records) {
        process(record);
        /** JVM executes business logic **/
    }
    consumer.commitSync();
    /** Consumer client sends commit to Group Coordinator **/
}
```

The Consumer Group Coordinator updates committed offsets.


```mermaid
sequenceDiagram
    participant P as Producer 
    participant B as Broker
    participant TC as Transaction Coordinator
    participant C as Consumer Client

    P->>B: Send message (idempotent)
    B->>P: Acks=all
    P->>TC: Commit transaction
    TC->>B: Make batch visible
    C->>B: Fetch committed messages
    B->>C: Deliver exactly-once messages
```

### When to Use Each Semantics

At Most Once: metrics, logs that can tolerate loss.
At Least Once: payment pipelines with deduplication in JVM layer.
Exactly Once: financial transactions, inventory updates, where duplicates cannot be tolerated.





Kafka **cannot guarantee exactly-once message delivery** at the transport level.
What Kafka guarantees is **exactly-once processing** (EOS) when the application follows specific patterns.

Meaning:

* A message may still be delivered more than once on the wire
* But it will be **processed exactly once** end-to-end (producer → broker → consumer → sink)

This is the *only* type of “exactly once” Kafka supports.
 

>To Achieve Kafka Exactly-Once Processing
>You must change **Producer**, **Broker**, and **Consumer** settings together.
 

###  Producer Settings for EOS

Enable idempotent producer and transactions.

Kafka ≥ 3.0 enables idempotence by default, but set these explicitly:

```
enable.idempotence=true
acks=all
retries=∞
max.in.flight.requests.per.connection=1
```

To turn on **transactions**, add:

```
transactional.id=my-producer-tx-id
```

The producer code must:

```
producer.initTransactions();
producer.beginTransaction();
producer.send(record);
producer.commitTransaction();
```

**This ensures every write happens exactly once even during retries.**
 
###  Broker Requirements

These are default in modern Kafka, but important:

```
unclean.leader.election.enable=false
min.insync.replicas=2    (if replication-factor ≥ 3)
```

Ensures **durable writes** and protects your EOS pipeline.



###  Consumer Requirements (Kafka Consumer w/o Streams)

Use the **transaction-aware consumer** settings:

```
isolation.level=read_committed
enable.auto.commit=false      (mandatory)
```

And your consumer must **manually commit offsets as part of the transaction** if you are producing to another topic.



###  End-to-End Exactly-Once Pattern (Consumer → Producer)

If your service reads from Kafka and writes back to Kafka, use a **transactional producer** to commit offsets atomically with output messages:

```
producer.beginTransaction();
Records = consumer.poll(...)
process(...)
producer.send(...)
producer.sendOffsetsToTransaction(consumerOffsets, consumerGroupId);
producer.commitTransaction();
```

This ensures:

* Offsets are committed **only if** output messages were produced
* If process fails, offsets roll back → input re-processed
* Re-processing does **not** produce duplicates
 

##  Using Kafka Streams

Kafka Streams automatically gives **exactly-once processing** by setting:

```
processing.guarantee=exactly_once_v2
```

That’s it. Streams manages transactions and offset commits for you.

 



You will **not** get EOS if:

* Consumer writes to an external system that doesn't support transactions
  (e.g., PostgreSQL without XA, Elasticsearch, MongoDB, Redis)
* You commit offsets before you finish processing
* You use auto-commit
* You do not use transactional producers
* You use multiple partitions that are not processed in-order

EOS requires the **full transactional chain**.

 

| Component         | Required Setting / Logic                                                                              |
| ----------------- | ----------------------------------------------------------------------------------------------------- |
| **Producer**      | `enable.idempotence=true`, `transactional.id`, `acks=all`, `beginTransaction() / commitTransaction()` |
| **Broker**        | `min.insync.replicas`, disabled unclean election                                                      |
| **Consumer**      | `isolation.level=read_committed`, manual commit, commit offsets in transaction                        |
| **Application**   | Atomic processing: read → process → write → commit offsets in same transaction                        |
| **Kafka Streams** | `processing.guarantee=exactly_once_v2`                                                                |




| Delivery Semantics     | Latency                                                                   | Reliability                                                              | Resource Usage                                                                                                       |
| ---------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------- |
| **At-most-once**       | **Lowest latency** (no retries, no confirmation waits)                    | **Lowest reliability** (messages may be lost; no retries; no guarantees) | **Lowest resource usage** (no duplicate checks, no transactions, minimal broker load)                                |
| **At-least-once**      | **Medium latency** (retries + acks + possible re-processing)              | **High reliability** (messages never lost; duplicates possible)          | **Medium resource usage** (extra bandwidth due to retries, more broker I/O, consumer re-processing)                  |
| **Exactly-once (EOS)** | **Highest latency** (transaction init, commit, coordination, more fsyncs) | **Highest reliability** (no loss + no duplicates end-to-end)             | **Highest resource usage** (transaction metadata, idempotence state, more CPU, memory, broker coordination overhead) |






 

More

* Full Java producer–consumer code for EOS
* Spring Kafka example with `KafkaTransactionManager`
* Kafka Streams EOS example
* A diagram of end-to-end transaction flow
* A detailed chart showing internal mechanisms for each semantic
* A Spring Kafka configuration comparison
* Java code examples for all three semantics
 

