# *Delivery Guarantees* in messaging systems 
 

# **Delivery Guarantee in Messaging Systems**

## **What It Means**

Delivery Guarantee defines **how a messaging system promises to deliver messages to consumers under failure conditions** such as:

* network issues
* broker crashes
* consumer restarts
* producer retries
* partition rebalancing
* acknowledgment loss

It describes the **contract** between **Producer → Broker → Consumer** about whether a message may be:

* delivered **once**
* delivered **multiple times**
* lost under failures
 

# **Why Delivery Guarantee Exists**

Messaging systems run over distributed networks where failures are *expected*, not rare.
Systems must choose:

* **availability**,
* **latency**,
* **durability**,
* **complexity**,
  based on business needs.

Delivery guarantees provide explicit behavior and expectations, so developers can design **idempotency**, **deduplication**, or **compensating logic** depending on the guarantee.

 

# **The Three Delivery Guarantee Models**

## **At-Most-Once**

* Messages delivered **0 or 1 time**
* No retries
* Fastest but may **lose messages**
* Used for logs, metrics, telemetry

**Contract:** No duplicates, but may drop messages.
 

## **At-Least-Once**

* Messages delivered **1 or more times**
* Retries enabled
* No message loss
* May deliver **duplicates**

**Contract:** Messages will not be lost, but must handle duplicate processing.
 

## **Exactly-Once**

* Delivered **once and only once**
* Requires coordination: transactions, idempotent writes, atomic commits
* Most expensive in latency/complexity
* Used for financial or inventory consistency

**Contract:** No loss, no duplicates — strict delivery semantics.

 

# **Delivery Guarantees Are System-Wide**

A messaging system’s delivery guarantee depends on:

* **Producer reliability**

  * acknowledgments, retries, batching
* **Broker reliability**

  * replication, durability, commit logs
* **Consumer reliability**

  * where/when offsets/acks are committed
* **Network behavior**

  * partitions, timeouts, lost ACKs
* **Storage durability**

  * replicated logs, WAL, sync writes

Even if the broker supports exactly-once, the **consumer application** must also behave correctly (idempotency, transactional commits) to truly achieve EOS.


 

| Guarantee         | Meaning              | Message Loss | Duplicates | Use Case                              |
| ----------------- | -------------------- | ------------ | ---------- | ------------------------------------- |
| **At-Most-Once**  | Deliver 0 or 1 times | Possible     | No         | logging, low-value telemetry          |
| **At-Least-Once** | Deliver ≥1 times     | No           | Possible   | orders, events, ingestion systems     |
| **Exactly-Once**  | Deliver exactly once | No           | No         | finance, payments, stateful pipelines |








## **At-Least-Once Delivery**

 

A message is delivered **one or more times**.
**Duplicates are possible**, loss is not.

### Why duplicates may occur

The producer sends a message → broker writes it → broker ACK gets **lost** → producer **retries** → message appended **again**.

### Kafka components causing behavior

* **Producer retries**
* **Broker replication delay**
* **Consumer committing offset after processing**

### Producer config enabling at-least-once

```properties
acks=all
retries=Integer.MAX_VALUE
enable.idempotence=false
```

### Consumer behavior

If consumer **commits offsets AFTER processing**, a retry may cause duplicate processing.

### Real-world example

Order service processes a payment **twice** because the first commit failed → duplicated events.

### Guarantee Summary

✔ No message loss
✖ Duplicate messages possible



## **At-Most-Once Delivery**

### Meaning

A message is delivered **zero or one time**.
**Loss is possible**, duplicates are not.

### Why messages can be lost

Consumer commits offset **before processing**.
If processing fails → offset already moved → record lost.

### Producer config

```properties
acks=0
retries=0
```

This means the producer does not wait for acknowledgements.

### Consumer behavior

Offset committed **before** processing.

### Real-world example

You receive **fewer logs** in ELK because some messages were dropped.

### Guarantee Summary

✔ No duplicates
✖ Potential message loss

---

# **Exactly-Once Delivery (EOS)**

### Meaning

A message is delivered **once and only once**, even in failures.

### Kafka components enabling EOS

1. **Idempotent Producer**

   * Ensures retries do not produce duplicates.
   * Broker deduplicates using **producerId + sequenceNumber**.

2. **Transactions API**

   * Producer writes multiple messages **atomically**.
   * Either **all writes** succeed or **none**.

3. **Consumer committed offsets in the same transaction** (if using Kafka Streams or transactions).

### Producer config for EOS

```properties
enable.idempotence=true
acks=all
retries=Integer.MAX_VALUE
transactional.id=my-txn-id
```

### Broker behavior

* Broker maintains a **per-producer sequence number** to prevent duplicates.
* Kafka’s transactional log ensures **atomic writes**.

### Consumer behavior

A consumer sees messages **only after the transaction is committed**.

### Real-world example

Banking system:
Debit from Account A AND credit to Account B must be an atomic event.
Kafka can guarantee this.

### Guarantee Summary

✔ No loss
✔ No duplicates
✔ Atomic multi-partition writes
✖ Slightly more latency due to transactions
 

 

# **Recommended Usage**

* **Always use At-least-once** unless duplicates break your business logic.
* Use **Exactly-once** only when doing:
  ✔ sink-to-Kafka-to-sink pipelines
  ✔ Kafka Streams
  ✔ financial consistency operations

More :
* Diagrams of retry cycles
* At-least vs at-most vs EOS flowchart
* Configurations for Kafka Streams EOS
* How consumer group commits affect guarantees

 

* Delivery guarantee diagram
* How each guarantee is implemented at low-level
* Differences across Kafka vs RabbitMQ vs SQS vs Pulsar
