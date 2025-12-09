### DynamoDB Terminology (With Clear, Concise Examples)

Below are **all important DynamoDB terms**, each with **2–3 line explanations** and **relevant examples**.

 

### Table

A collection of items (similar to a table in relational DB) but **schema-less** except for PK/SK.

**Example:**
A table named `Ecommerce` storing Users, Orders, Products—all in one table.

 

### Item

A single record inside a DynamoDB table.

**Example:**

```
{ "PK": "USER#1", "SK": "USER#1", "Name": "John" }
```

 

### Attribute

A field inside an item. DynamoDB allows **string, number, map, list, binary**, etc.

**Example:**
`Name`, `Age`, `Address.zip` are attributes.

 

### Primary Key (PK)

Uniquely identifies an item. Two types:

1. **Partition Key only**
2. **Partition Key + Sort Key**

**Example:**
`PK = USER#1`.

 

### Partition Key (Hash Key)

Determines the **physical partition** the item goes into.
Items with the same PK are grouped together.

**Example:**
All `PK = USER#1` items stored contiguously.

 

### Sort Key (Range Key)

Defines **ordering** inside a partition.

**Example:**

```
PK = USER#1
SK = ORDER#202501
```

 

### Composite Key

PK + SK together uniquely identify an item.

**Example:**

```
PK = USER#1
SK = ORDER#1001
```

 

### Partition

Underlying storage units managed by DynamoDB to distribute data and workload.

**Example:**
If you have heavy writes, DynamoDB automatically splits your partitions.

 

### Secondary Index (GSI & LSI)

Alternate key structures for additional query patterns.

 

### Global Secondary Index (GSI)

Index with **different PK and SK** from the main table.

**Example:** Query users by email:

```
GSI1PK = EMAIL#john@example.com
GSI1SK = USER#1
```

 

### Local Secondary Index (LSI)

Shares the same **PK**, but uses a **different SK**.

**Example:**
Sort orders by `OrderDate` instead of `SK`.

 

### Provisioned Capacity

You configure **RCU** and **WCU** manually. Good for predictable traffic.

 

### On-Demand Capacity

DynamoDB auto-scales capacity. Best for unpredictable traffic.

 

### Read Capacity Unit (RCU)

1 strongly consistent read of 4 KB per second.

**Example:** Reading an item of 8 KB = **2 RCUs**.

 

### Write Capacity Unit (WCU)

1 write of 1 KB per second.

**Example:** Writing a 2 KB item = **2 WCUs**.

 

### Strongly Consistent Read

Reads the **latest** committed value.

 

### Eventually Consistent Read

May return stale data but is **2× cheaper**.

 

### Query

Fetches items by **PK** and optional **SK condition**.

**Example:**
Get all orders of user:

```
PK = USER#1
SK begins_with ORDER#
```

 

### Scan

Reads **every item** in the table or index. Costly.

**Example:**
Scan table to find all products with price < 100.

 

### Filter Expression

Filters results **after** query/scan—extra cost.

**Example:**
`price < 100` applied after reading items.

 

### Projection Expression

Selects only specific attributes to reduce cost.

**Example:**
`Name, Email`.

 

### Expression Attribute Names

Used when attribute names conflict with reserved keywords.

**Example:**
`#name = "John"` where `#name` refers to the Name attribute.

 

### Expression Attribute Values

Placeholders for attribute values.

**Example:**
`:status = "PAID"`.

 

### TTL (Time To Live)

Automatically deletes items after a timestamp.

**Example:**
`ttl = 1700000000` (expires at this UNIX time).

 

### DynamoDB Streams

Captures **real-time changes** (insert, update, delete) for processing via **AWS Lambda**.

 

### BatchGetItem

Reads up to **100 items** across multiple tables in a single request.

 

### BatchWriteItem

Writes/deletes up to **25 items** in one request.

 

### Conditional Writes

Writes occur only if a condition is satisfied.

**Example:**
Only deduct inventory if `stock > 0`.

 

### Optimistic Concurrency Control (OCC)

Uses a version attribute to prevent overwriting newer updates.

 

### Item Collection

All items with the same PK.

**Example:**
All `USER#1` items (orders, addresses, etc.).

 

### Single Table Design (STD)

Storing multiple entity types in one table using PK/SK strategies.

 

### Hot Partition

A partition receiving disproportionate traffic → throttle.
DynamoDB uses internal auto-splitting to reduce this.

 

### Throttling

Occurs when RCU/WCU limits exceed. SDK returns `ProvisionedThroughputExceededException`.

 

### Auto Scaling

DynamoDB automatically adjusts RCU/WCU based on demand.

 

### S3 Export / Import

Move DynamoDB data to/from S3 without writing custom code.

 

### Transactions

ACID-compliant operations across multiple items.

**Example:**
Decrease inventory **and** increase order count atomically.

 

If you want, I can prepare a **one-page cheat sheet table (markdown)** with all DynamoDB terms, examples, and purpose.


To design DynamoDB for globally distributed applications requiring low latency and strong consistency, consider these best practices and trade-offs:

1. **Global Tables for Multi-Region Replication:**
   - Use DynamoDB Global Tables to replicate tables automatically across multiple AWS regions.
   - This allows active-active writes in all regions, improving latency by serving users from the closest region.
   - Ensures high availability and disaster recovery by maintaining data presence in multiple regions.

2. **Conflict Resolution:**
   - DynamoDB global tables use a last-writer-wins model based on timestamps to resolve write conflicts.
   - Application logic should accommodate potential inconsistencies, especially where concurrent writes occur.

3. **Consistency Models and Data Integrity:**
   - Global tables offer eventual consistency across regions for replicated data.
   - For strongly consistent reads, queries must be directed to the region that processes the write.
   - Balance between latency-sensitive reads and consistency requirements is critical.

4. **Capacity and Cost Optimization:**
   - Provision or use on-demand capacity mode equally across all global table replicas to avoid replication capacity bottlenecks.
   - Use auto-scaling to adjust capacity automatically based on workload variations.
   - Monitor replication lag via CloudWatch metrics to ensure performance SLAs.

5. **Latency and Performance Optimization:**
   - Select regions geographically close to users to reduce latency.
   - Partition keys should be designed to evenly distribute traffic and avoid hot partitions in all regions.

6. **Data Locality and Compliance:**
   - Use multi-region replication to meet compliance and data residency requirements by controlling where data is stored.

7. **Disaster Recovery and Failover:**
   - Test failover scenarios to verify that applications continue functioning seamlessly if a region becomes unavailable.

In summary, DynamoDB Global Tables support building globally available, low-latency applications, but require careful planning around conflict resolution, capacity management, consistency trade-offs, and region selection to optimize performance, cost, and data integrity[1][4][5][6][7][8].

Citations:
[1] Using DynamoDB Global Tables for Multi-Region ... https://dev.to/imsushant12/using-dynamodb-global-tables-for-multi-region-applications-ml8
[2] Understanding and Exploring Global Tables on Dynamodb https://notes.kodekloud.com/docs/AWS-Certified-SysOps-Administrator-Associate/Domain-2-Reliability-and-BCP/Understanding-and-Exploring-Global-Tables-on-Dynamodb
[3] Designing Highly Available Architectures with DynamoDB https://www.valuebound.com/resources/blog/designing-highly-available-architectures-dynamodb
[4] Best practices for global tables - Amazon DynamoDB https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/globaltables-bestpractices.html
[5] Best practices and requirements for managing global tables https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/globaltables_reqs_bestpractices.html
[6] Multi-Region Design Amazon DynamoDB Global Tables https://www.youtube.com/watch?v=tgWNkUHJgPU
[7] Amazon DynamoDB Best Practices: 10 Tips to Maximize ... https://www.simform.com/blog/dynamodb-best-practices/
[8] Best practices for designing and architecting with DynamoDB https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/best-practices.html
[9] DynamoDB Data Modeling Best Practices - AWS - Amazon.com https://aws.amazon.com/awstv/watch/7dbdede0b17/


Great question — and this is exactly the trap many people fall into in interviews.
On the surface, it looks like a hot partition should only affect that partition — but in DynamoDB, that’s not true for several very real reasons.

Here is the precise explanation that AWS expects:


 

✅ Why does a hot partition (noisy neighbor) impact other tenants?

Although DynamoDB partitions are isolated for storage, they share table-level throughput and backend resources.
So a single hot partition can affect the entire table in three ways:


 

1️⃣ Table-level WCU/RCU is shared across all partitions

Even though each partition has a physical limit (around 3,000 RCUs / 1,000 WCUs per partition),
the entire table’s provisioned capacity is shared across all partitions.

Example:

Table provisioned write capacity = 10,000 WCUs

Tenant A (noisy neighbor) uses 1 partition and bursts to 8,000 WCUs

Remaining 9 partitions must now share only 2,000 WCUs


➡ Other tenants get throttled even though their partitions are not hot.

This is the #1 cause of cross-tenant impact.


 

2️⃣ Adaptive Capacity helps, but only AFTER throttling starts

DynamoDB can move unused capacity to hot partitions through Adaptive Capacity,
but:

It reacts after throttling begins

It reallocates only unused table-level capacity

It cannot exceed hard per-partition limits (like ~1,000 WCUs)


➡ So if the noisy tenant exhausts the table's unused capacity, others see throttling before adaptive capacity stabilizes.


 

3️⃣ Internal DynamoDB resources are shared (workers, request queues, partitions placement)

Internally, DynamoDB uses:

Shared request queues

Shared worker pools

Shared storage nodes hosting multiple partitions


A single tenant doing extremely high request rates:

Fills the request queues

Increases compaction/replication work on its partitions

Can cause queue backpressure for all partitions on the same partition server


➡ Result: Increased latency + throttling for unrelated partitions.

This is the fundamental “noisy neighbor” problem.

Even AWS re:Invent talks explicitly say:
"One hot partition can impact the performance of the entire table."


 

4️⃣ GSIs make it worse (hot GSI partitions = global throttling)

If a noisy neighbor causes hot activity on a GSI:

GSIs have their own partitions

GSIs share their own throughput limits


A single tenant whose items update a GSI heavily causes:

GSI write throttles

Which then throttle writes on the main table


➡ This cascades throttling to everyone.


 

5️⃣ Large spikes cause request-routing imbalances at the front-end fleet

DynamoDB’s front-end routers distribute requests across backend partitions.
A sudden 100× spike from a single tenant:

Floods the routing tier

Causes queuing delays for other tenants

Returns throttling (ProvisionedThroughputExceeded)



 

🧠 Simplified answer for interviews

Use this version in interviews:

> “Hot partitions affect the table globally because throughput is allocated at the table level, not per partition.
A single noisy neighbor can consume most WCUs/RCUs and fill shared internal request queues, causing throttling and high latency for others. Adaptive capacity helps but reacts only after throttling, and it can’t break the hard per-partition limits.
Therefore, a hot partition always risks degrading performance for the entire table.”




 

If you want, I can also explain:

✔ How DynamoDB internally balances partitions
✔ How many partitions a table will have
✔ How adaptive capacity actually redistributes throughput internally
✔ How to completely isolate tenants using dedicated tables or capacity units

Want the deeper internal mechanics?


Great follow-up — this is exactly where DynamoDB design clicks.
Let’s go deeper and make it crystal clear.


 

✅ Why adding a shardId to the partition key helps (even though the table has limited partitions)

Adding a shard ID dramatically increases the number of distinct partition key values.
DynamoDB will then split these keys across many physical partitions, giving you:

✔ More partitions

✔ More throughput

✔ No hot key concentration

✔ Adaptive capacity becomes far more effective

Let’s break it down.


 

✅ 1) DynamoDB partitions are created based on key cardinality + throughput pressure

A table does not have a fixed number of partitions.
DynamoDB automatically creates new partitions whenever:

1. Storage grows near 10GB


2. A partition hits throughput limits (WCU/RCU pressure)



When you add a shardId (synthetic sharding), example:

PK = tenantId#00
PK = tenantId#01
PK = tenantId#02
…
PK = tenantId#09

Suddenly the table has 10× more partition key values, so DynamoDB spreads these logical keys across more physical partitions.

➡ More physical partitions means more total available throughput.


 

✅ 2) Writes get evenly distributed across partitions instead of hitting one partition

Without sharding:

PK = tenantA
All writes → same partition → throttling

With sharding:

PK = tenantA#0 → Partition #37
PK = tenantA#1 → Partition #12
PK = tenantA#2 → Partition #89
...

Now your workload is spread across multiple backend partitions.

➡ Instead of 1 partition hitting 1,000 WCUs limit, you now have 10 partitions → ~10,000 WCUs potential capacity.


 

✅ 3) Adaptive capacity can now work properly

Adaptive capacity can move unused throughput across partitions only when there is high key cardinality.

If you have one partition key:

Adaptive capacity can’t help

All requests go to the same backend node → hot partition → throttles


If you have 10–20 sharded keys:

Adaptive capacity reallocates throughput between backend partitions

Hotter shards get more throughput

No throttles to other tenants



 

✅ 4) Even though the table is “one logical table,” partitions scale up underneath it

A DynamoDB table is logically one table, but physically:

Table
  ├─ Partition 1
  ├─ Partition 2
  ├─ Partition 3
  ├─ ...
  └─ Partition N  (created automatically)

Sharding increases N.

More partitions = More write queues, more storage isolation, more request capacity.


 

✅ 5) Adding a shardId multiplies the per-tenant maximum throughput

Let’s use numbers — they make this extremely obvious:

Without shard ID

Tenant A writes 2,000 WCU/sec

One partition can do ~1,000 WCU

→ Throttling


With shard ID (10 shards)

Tenant A writes 2,000 WCU/sec

Writes distributed across 10 partitions (~200 per shard)

Each shard well below 1,000 WCU limit

→ NO throttling


Sharding takes you from a physics limit (1 partition ≈ 1,000 WCUs)
to a scalable model (N partitions ≈ N * 1,000 WCUs).


 

✔ Concise Interview Answer

You can answer in this clean short form:

> “A hot partition forms when too many writes hit the same partition key. Adding a shardId creates multiple partition key values, which DynamoDB distributes across multiple physical partitions.

So instead of one key overwhelming one partition (≈1000 WCUs limit), the workload spreads across many backend partitions. This increases total throughput, removes the single hot spot, and allows adaptive capacity to redistribute unused throughput.

That’s why synthetic sharding eliminates hot partitions.”




 

If you want, I can also explain:

How to choose the right number of shards (formula)

Automatic shard scaling pattern

How to shard GSIs safely

When to move a tenant to a dedicated table instead of sharding


Just tell me.


Below is the most accurate and interview-ready summary of DynamoDB RCU/WCU limits at every level:
table, partition, GSI, LSI, and account-level.

This is a topic AWS loves asking because people confuse the “per-partition” vs “per-table” limits.


 

✅ 1. Partition-Level Limits (hard physical limits)

These limits are non-negotiable — the most important to understand.

For each physical partition:

Resource	Limit

Write throughput	~1,000 WCUs per partition
Read throughput	~3,000 RCUs per partition
Storage	10 GB max per partition


These limits are why hot partitions cause throttling.
No matter how much table capacity you provision, each partition cannot exceed these numbers.


 

✅ 2. Table-Level Throughput Limits (soft / scalable)

A table’s total throughput = sum of all its physical partitions.

DynamoDB will automatically create more partitions when:

1. Table size nears 10 GB × #partitions


2. Provisioned throughput increases enough


3. Traffic patterns require repartitioning



Table has NO global throughput ceiling except your AWS account limits.

You can provision:

Hundreds of thousands of WCUs

Hundreds of thousands of RCUs

Table will scale partitions accordingly.


But a single partition key stays limited to the partition-level limits above.


 

✅ 3. GSI-Level Throughput Limits

Each GSI is its own index with its own partitions, so:

GSI partitions have the SAME limits as a table:

Resource	Limit

Write throughput	~1,000 WCUs per partition
Read throughput	~3,000 RCUs per partition
Storage	10 GB per partition


BUT two important constraints:

A) GSI write throttling throttles the base table

Whenever you write to the base table, DynamoDB also writes to the GSI.
If the GSI is throttled → the base table write is throttled.

B) GSI scaling is independent of table scaling

GSIs scale their partitions separately from table partitions.


 

✅ 4. LSI-Level Limits (Local Secondary Index)

LSIs share partitions with the base table.
Therefore:

LSIs have the EXACT SAME partition limits as the base table

Because LSIs reuse the base table’s storage nodes.

Meaning:

~1,000 WCUs / partition

~3,000 RCUs / partition

10 GB storage per partition TOTAL (base + LSI indexes combined)


LSIs cannot exceed these limits or scale independently.


 

✅ 5. Item-Level Limits (not throughput, but important)

Limitation	Value

Max item size	400 KB
Strongly consistent read	1 RCU = 4KB read
Eventually consistent read	1 RCU = 8KB read
1 WCU writes (up to 1KB)	1 KB per WCU



 

✅ 6. Account-Level Limits (Soft Limits)

Per AWS account per region:

Resource	Default Soft Limit	Hard Limit

Provisioned WCUs	~40k	Scales higher with request
Provisioned RCUs	~80k	Scales higher
On-Demand Burst	Region-dependent	Flexible
Number of tables	~256	Request increase


These are easy to increase with AWS Support.


 

🔥 Putting Everything Together (Exam Summary)

Use this 5-liner during an interview:

> “DynamoDB partitions have hard limits: ~1,000 WCUs, ~3,000 RCUs, and 10 GB per partition.
A table consists of many such partitions, so table throughput has no fixed upper limit.
GSIs also have their own partitions with the same per-partition limits.
LSIs share partitions with the base table, so they share the same limits and cannot scale independently.
Account-level limits cap the maximum provisioned capacity unless increased by AWS Support.”




 

Want a diagram?

I can create a diagram showing the hierarchy of RCU/WCU limits like:

AWS Account
   └── Table
         ├── Partitions
         │      ├── 1000 WCUs max
         │      ├── 3000 RCUs max
         │      └── 10GB storage
         ├── GSIs → own partitions
         └── LSIs → shared partitions

Should I create the diagram for you?
