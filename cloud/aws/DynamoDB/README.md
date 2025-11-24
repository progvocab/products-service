### ### DynamoDB Terminology (With Clear, Concise Examples)

Below are **all important DynamoDB terms**, each with **2–3 line explanations** and **relevant examples**.

---

### ### Table

A collection of items (similar to a table in relational DB) but **schema-less** except for PK/SK.

**Example:**
A table named `Ecommerce` storing Users, Orders, Products—all in one table.

---

### ### Item

A single record inside a DynamoDB table.

**Example:**

```
{ "PK": "USER#1", "SK": "USER#1", "Name": "John" }
```

---

### ### Attribute

A field inside an item. DynamoDB allows **string, number, map, list, binary**, etc.

**Example:**
`Name`, `Age`, `Address.zip` are attributes.

---

### ### Primary Key (PK)

Uniquely identifies an item. Two types:

1. **Partition Key only**
2. **Partition Key + Sort Key**

**Example:**
`PK = USER#1`.

---

### ### Partition Key (Hash Key)

Determines the **physical partition** the item goes into.
Items with the same PK are grouped together.

**Example:**
All `PK = USER#1` items stored contiguously.

---

### ### Sort Key (Range Key)

Defines **ordering** inside a partition.

**Example:**

```
PK = USER#1
SK = ORDER#202501
```

---

### ### Composite Key

PK + SK together uniquely identify an item.

**Example:**

```
PK = USER#1
SK = ORDER#1001
```

---

### ### Partition

Underlying storage units managed by DynamoDB to distribute data and workload.

**Example:**
If you have heavy writes, DynamoDB automatically splits your partitions.

---

### ### Secondary Index (GSI & LSI)

Alternate key structures for additional query patterns.

---

### ### Global Secondary Index (GSI)

Index with **different PK and SK** from the main table.

**Example:** Query users by email:

```
GSI1PK = EMAIL#john@example.com
GSI1SK = USER#1
```

---

### ### Local Secondary Index (LSI)

Shares the same **PK**, but uses a **different SK**.

**Example:**
Sort orders by `OrderDate` instead of `SK`.

---

### ### Provisioned Capacity

You configure **RCU** and **WCU** manually. Good for predictable traffic.

---

### ### On-Demand Capacity

DynamoDB auto-scales capacity. Best for unpredictable traffic.

---

### ### Read Capacity Unit (RCU)

1 strongly consistent read of 4 KB per second.

**Example:** Reading an item of 8 KB = **2 RCUs**.

---

### ### Write Capacity Unit (WCU)

1 write of 1 KB per second.

**Example:** Writing a 2 KB item = **2 WCUs**.

---

### ### Strongly Consistent Read

Reads the **latest** committed value.

---

### ### Eventually Consistent Read

May return stale data but is **2× cheaper**.

---

### ### Query

Fetches items by **PK** and optional **SK condition**.

**Example:**
Get all orders of user:

```
PK = USER#1
SK begins_with ORDER#
```

---

### ### Scan

Reads **every item** in the table or index. Costly.

**Example:**
Scan table to find all products with price < 100.

---

### ### Filter Expression

Filters results **after** query/scan—extra cost.

**Example:**
`price < 100` applied after reading items.

---

### ### Projection Expression

Selects only specific attributes to reduce cost.

**Example:**
`Name, Email`.

---

### ### Expression Attribute Names

Used when attribute names conflict with reserved keywords.

**Example:**
`#name = "John"` where `#name` refers to the Name attribute.

---

### ### Expression Attribute Values

Placeholders for attribute values.

**Example:**
`:status = "PAID"`.

---

### ### TTL (Time To Live)

Automatically deletes items after a timestamp.

**Example:**
`ttl = 1700000000` (expires at this UNIX time).

---

### ### DynamoDB Streams

Captures **real-time changes** (insert, update, delete) for processing via **AWS Lambda**.

---

### ### BatchGetItem

Reads up to **100 items** across multiple tables in a single request.

---

### ### BatchWriteItem

Writes/deletes up to **25 items** in one request.

---

### ### Conditional Writes

Writes occur only if a condition is satisfied.

**Example:**
Only deduct inventory if `stock > 0`.

---

### ### Optimistic Concurrency Control (OCC)

Uses a version attribute to prevent overwriting newer updates.

---

### ### Item Collection

All items with the same PK.

**Example:**
All `USER#1` items (orders, addresses, etc.).

---

### ### Single Table Design (STD)

Storing multiple entity types in one table using PK/SK strategies.

---

### ### Hot Partition

A partition receiving disproportionate traffic → throttle.
DynamoDB uses internal auto-splitting to reduce this.

---

### ### Throttling

Occurs when RCU/WCU limits exceed. SDK returns `ProvisionedThroughputExceededException`.

---

### ### Auto Scaling

DynamoDB automatically adjusts RCU/WCU based on demand.

---

### ### S3 Export / Import

Move DynamoDB data to/from S3 without writing custom code.

---

### ### Transactions

ACID-compliant operations across multiple items.

**Example:**
Decrease inventory **and** increase order count atomically.

---

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
