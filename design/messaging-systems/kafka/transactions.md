

#    Kafka Transactions?

Kafka transactions allow a producer to **group multiple writes (and offset commits) into a single atomic unit**:

* Either **all** writes succeed together
* Or **none** of them are visible to consumers

This is the foundation for **exactly-once processing (EOS)** in Kafka.


###   Why Do We Need Kafka Transactions?

Without transactions:

* Producer retries may create duplicates
* Consumers may see partial results
* A common issue: **processed message gets written to an output topic but consumer offset commit fails → duplicate processing

Transactions fix this by allowing:

1. **Atomic write of output messages**
2. **Atomic commit of input-message offsets**

So the system (read → process → write) becomes **exactly once**.


###   Core Plumbing Behind Kafka Transactions

Kafka uses a **transaction coordinator** running inside the broker.

When a producer declares a transactional ID (e.g., `"payment-tx-1"`), Kafka:

1. Assigns the producer to a **transaction coordinator
2. Creates a **transaction log** internal topic: `__transaction_state`
3. Tracks:

   * Open transactions
   * Producer epoch & sequence numbers
   * Partitions touched by the transaction


###   Transaction Workflow

#### **1. initTransactions()**

Producer registers its `transactional.id` with the transaction coordinator.
This also fences off old zombie producers with the same ID.

#### **2. beginTransaction()**

Producer starts a new transactional batch.
All records sent now are in a pending state.

#### **3. send(record)**

Records are produced to partitions but are tagged as part of an **ongoing transaction**.
Consumers do **not** see them unless they use:

```
isolation.level=read_committed
```

#### **4. Optionally send offsets into the transaction**

If doing consume → process → produce:

```
producer.sendOffsetsToTransaction(offsets, consumerGroupId)
```

This is the important piece that commits **input offsets atomically** with output messages.

#### **5. commitTransaction()**

The transaction coordinator writes a **COMMIT** marker to the log.
All records become **visible** to read_committed consumers.

#### **If Something Fails → abortTransaction()**

Writes an **ABORT** marker.
All pending messages are discarded.


###   What Happens Internally on Commit

1. Producer sends **EndTransaction** to coordinator
2. Coordinator writes COMMIT/ABORT markers to `__transaction_state`
3. Coordinator sends transactional markers to all partitions touched
4. Each partition appends a **commit marker**
5. Consumer with `read_committed`:

   * Hides uncommitted records
   * Returns only committed ones

This is how Kafka creates isolation similar to **database transactions**, but optimized for distributed logs.


###   Transaction Guarantees

Kafka transactions offer:

| Guarantee                   | Explanation                                     |
| --------------------------- | ----------------------------------------------- |
| **Atomicity**               | All writes & offset commits succeed as one unit |
| **Isolation**               | Consumers never see partial results             |
| **Idempotence**             | Retries don't generate duplicates               |
| **Exactly-once processing** | Possible for read → process → write pipelines   |


###   How Transactions Enable Exactly-Once Processing

To achieve EOS, you need:

1. **Idempotent producer**
2. **Transactional writes**
3. **Transactional offset commits**
4. **read_committed consumers**

This ensures:

* If the app crashes before commit → offsets not committed → message reprocessed
* But duplicates will not be written because the whole transaction aborted


###   What Transactions Do NOT Do

Kafka transactions **cannot**:

* Guarantee exactly-once delivery to external systems like DB, Elasticsearch, Mongo (unless wrapped in 2PC)
* Work across multiple independent producers unless carefully coordinated
* Give high throughput without tuning (EOS has overhead)


###   When Should You Use Kafka Transactions?

Use them only when you need real **exactly-once processing**, such as:

* Payment systems
* Inventory decrement + event generation
* Stream processing (state updates)
* ETL: read → enrich → publish to next topic

Don’t use transactions when:

* Throughput is more important than EOS
* Consumers can handle duplicates
* Data is immutable and idempotent (logs, analytics)

More

* Java transactional producer example
* Spring Kafka EOS example
* Kafka Streams EOS code
* Architecture diagram of the transaction flow

Tell me what you'd like next.

