
You have:

* **WCU = 500**
* **RCU = 1500 (eventual consistency → each strongly-consistent RCU = 2 eventual, but not relevant to writes)**
* **Table size = 10 GB → 1 partition**
* **No Auto Scaling**
* **Write size = 1 KB → 1 write = 1 WCU**
* **Write load = 1000 writes/sec**

Now let’s walk through what DynamoDB (internal components: *partition router, burst credit engine, adaptive capacity unit*) does.

---

### ### Step 1 — You receive 1000 writes/sec

→ **Demand = 1000 WCU/sec**
→ **Provisioned = 500 WCU/sec**
→ **Deficit = 500 WCU/sec**

---

### ### Step 2 — DynamoDB serves up to 500 writes/sec normally

The **partition** can serve only the provisioned throughput:

```
500 writes/sec served immediately
```

Remaining:

```
500 writes/sec over limit
```

---

### ### Step 3 — DynamoDB tries to use **burst capacity**

Each partition accumulates **burst credits** equal to **5 minutes of unused capacity**.

* If you were idle or underutilized earlier → you have credits.
* If you recently used full capacity → credits are low.

So DynamoDB will attempt:

```
Use burst credits to serve some or all of the extra 500 writes/sec
```

This is correct.

---

### ### Step 4 — If burst credits are insufficient

Then DynamoDB evaluates **adaptive capacity**:

* Adaptive capacity can *reallocate unused capacity across partitions*.
* But you have **only 1 partition**.
* With 1 partition, **there is no other partition to borrow from**.

So **adaptive capacity = no effect**.

This is also correct.

---

### ### Step 5 — The remaining writes are throttled

Once provisioned capacity + burst credits are exhausted:

```
Remaining writes → ProvisionedThroughputExceededException
```

Or in the response metadata (for batch operations):

```
Throttled: true
```

So your conclusion is correct:

✔ Only 500 writes/sec guaranteed
✔ Some additional writes served from burst credits
✔ No adaptive capacity help (1 partition only)
✔ Remaining writes are throttled

---

# ✔ Final Correct Statement

Here is the precise, correct final answer:

> With 500 WCU, 1 KB writes, and a 10 GB table (1 partition), DynamoDB serves **500 writes/sec** normally.
>
> It then uses **burst capacity** to try to serve additional writes.
>
> Because there is **only one partition**, DynamoDB **cannot borrow capacity** from any other partition.
>
> Once burst credits drain, all remaining writes (beyond 500/sec) are **throttled**.

This is exactly how DynamoDB behaves.

---

If you want, I can also explain:

* What happens when you increase WCU or storage (partition splits)
* How On-Demand mode handles the same 1000 writes/sec scenario
* How to redesign the table with sharding to support future scaling




### Key facts (from AWS)

* A single DynamoDB partition is designed to support up to **3,000 RCUs** and **1,000 WCUs**. ([AWS Documentation][1])
* DynamoDB allocates partitions to meet **the greater of** (a) throughput needs (RCU/WCU) and (b) storage needs (≈10 GB per partition). ([AWS Documentation][2])

### Calculation for your table (500 WCU, 1500 RCU)

1. Required partitions for **writes** = `ceil(500 WCU / 1000 WCU per partition)` = `ceil(0.5)` = **1 partition**. ([AWS Documentation][1])
2. Required partitions for **reads** = `ceil(1500 RCU / 3000 RCU per partition)` = `ceil(0.5)` = **1 partition**. ([AWS Documentation][1])
3. Required partitions for **storage** = `ceil(table_size_bytes / 10 GB)`. (If your table size is small, this is 1.) ([AWS Documentation][2])

### Answer

* **Initial partition count = 1** (because both read and write requirements fit into a single partition and assuming table storage < 10 GB).

### Caveats and practical notes

* AWS may allocate extra partitions internally for operational headroom; the calculcation above is the expected minimum based on published per-partition throughput. ([AWS Documentation][2])
* If you add GSIs, each index has its own partitions and capacity considerations (GSIs can increase total partition count). ([AWS Documentation][3])
* Hot-partition behavior, adaptive capacity, and burst credits affect observed throttling but do **not** change the basic partition-count math above. ([AWS Documentation][3])

If you’d like, I can compute expected partitions including a given table size (GB) or show how GSIs would change the partition count.

[1]: https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/bp-partition-key-design.html?utm_source=chatgpt.com "Best practices for designing and using partition keys ..."
[2]: https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/HowItWorks.Partitions.html?utm_source=chatgpt.com "Partitions and data distribution in DynamoDB"
[3]: https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/burst-adaptive-capacity.html?utm_source=chatgpt.com "DynamoDB burst and adaptive capacity"


Short answer: **The number of partitions does NOT change just because you chose `employeeId` as the partition key and `departmentId` as the sort key.**
Partition count depends ONLY on **provisioned capacity** and **table size**, not on the key names.

Let’s break it down clearly.

---

# ✅ **1. Partition Count = MAX(Throughput-Based Partitions, Size-Based Partitions)**

### ✔ From your previous example:

* **WCU = 500**
* **RCU = 1500**

### **Throughput-based calculation**

* 1 partition supports:

  * **1000 WCU**
  * **3000 RCU**

So:

* Required write partitions = `500 / 1000 = 0.5 → 1`
* Required read partitions = `1500 / 3000 = 0.5 → 1`

👉 Throughput-based partitions = **1**

### **Size-based calculation**

* 1 partition can store **up to 10 GB** of data.

Since table size is not mentioned, we assume it's **< 10 GB** → **1 partition**

### **Final partition count = max(1, 1) = 1**

---

# ✅ **2. Does changing PK → `employeeId` AND SK → `departmentId` change partition count?**

**No.**

The number of partitions is NOT affected by:

* The name of the partition key
* The type of the partition key
* Adding or removing a sort key

Partition count is affected only by:

### **✔ Provisioned RCU/WCU**

### **✔ Total data size stored in the table**

### ❌ NOT affected by logical key design

---

# 🧠 But here’s the important part...

### Changing **what** you use as the partition key **does affect the distribution of items across partitions**, even if the **number** of partitions stays the same.

If `employeeId` is evenly distributed, then writes/reads spread nicely.

If `employeeId` is skewed (e.g., some IDs very hot), you can still experience:

* **Hot partition**
* **Hot key throttling**
* **Imbalanced throughput**

But the **partition count remains 1** until:

* You exceed **1000 WCU** (write split)
* You exceed **3000 RCU** (read split)
* You exceed **10 GB** (storage split)

Then DynamoDB automatically creates **additional partitions**, but still does not care what your key *name* is.

---

# 📌 Final Answer

**Even after setting `employeeId` as PK and `departmentId` as SK, the number of partitions remains exactly the same: 1.**

Partition key attributes affect **distribution**, not **partition count**.

---

If you want, I can also explain:

* When partitions split dynamically
* What happens during autoscaling
* Hot key/hot partition examples using an Employees table
* How to redesign keys to avoid throttling

Just tell me!


Adding a **shard ID** (or "bucket ID") into the **partition key** does **NOT increase the number of DynamoDB physical partitions**, but it **changes how your items are *distributed* across existing partitions**, which dramatically improves throughput distribution and avoids hot partitions.

Let’s break this down cleanly.

---

# ✅ **1. Physical Partition Count ≠ Logical Key Design**

DynamoDB physical partition count is determined ONLY by:

### ✔ Provisioned WCU & RCU

### ✔ Item size / total storage

### ❌ NOT by how you design the partition key (e.g., adding a shard ID)

So adding `shardId` does *not* directly increase physical partitions.

---

# ✅ **2. But Adding a Shard ID Changes DISTRIBUTION of Keys**

Normally, this is your key:

```
PK = employeeId
SK = eventTimestamp
```

If many writes come for the same `employeeId`, then **all writes hit a single physical partition** → **hot partition** → throttling.

If you change it to:

```
PK = shardId#employeeId
SK = eventTimestamp
```

where `shardId` is a number (0–9 for 10 shards):

* Employee 123 with shard 0 → `0#123`
* Employee 123 with shard 1 → `1#123`
* etc.

Now the **same logical employee is spread across multiple partition-key values**, allowing DynamoDB to distribute writes across multiple internal partitions (once the table has >1 partition).

---

# ⚙️ **3. Why Does DynamoDB Spread the Load Better?**

DynamoDB hashes your partition key:

```
internalPartitionNumber = HASH(partitionKey) % numberOfPhysicalPartitions
```

Before using shardId:

* PK: `employee123`
* Only one hash → always mapped to **one** physical partition
  → **Hot partition** if high write rate.

After using shardId:

* PKs: `0#123`, `1#123`, `2#123` …
* Each produces **different hash value** → spread across **multiple physical partitions**

Even if physical partition count stays same, the *hash randomization* improves distribution.

Once autoscaling adds more partitions (due to high WCU/RCU), the shard IDs start spreading load evenly.

---

# 📌 **4. When does shard ID actually help?**

Shard IDs are useful only when:

### ✔ Your table has multiple physical partitions

(either due to size >10GB or WCU>1000 or RCU>3000)

### ✔ You have hot keys (e.g., same PK receives heavy traffic)

Shard IDs allow DynamoDB to distribute that load across partitions.

---

# 🧠 **5. How many shards should I create?**

General rules of thumb:

| Write rate per PK | Recommended Shards |
| ----------------- | ------------------ |
| 1k writes/sec     | 5–10               |
| 5k writes/sec     | 20–50              |
| 10k+ writes/sec   | 50–100             |

More shards → finer load distribution.

---

# 🔥 **6. Practical Example: Employees Table**

**Before (Hot Key Problem)**

```
PK = employeeId = 123
1000 writes/sec for employee 123
```

All writes hit ONE DynamoDB partition → throttling.

---

**After (Sharded Design)**

```
PK = shardId#employeeId
(shardId ranges from 0 to 9 → 10 shards)
```

Now:

```
0#123 → 100 writes/sec
1#123 → 100 writes/sec
...
9#123 → 100 writes/sec
```

Each flows to a different internal partition.

---

# 🎯 **7. FINAL ANSWER**

Adding a shard ID in the partition key:

### ✔ Does *not* change the number of physical partitions

### ✔ But changes the **distribution** of items across partitions

### ✔ Enables high write throughput without hot partitions

### ✔ Uses hashing to spread load across *existing* and *future* partitions

### ✔ Is a common design pattern for high-traffic DynamoDB tables

This is why **shard ID + autoscaling** is the most scalable DynamoDB design pattern.

---

If you want, I can also show:

* How to query sharded data efficiently
* How to combine shards for consistent reads
* A complete table design with GSIs
* Code examples (Java/Python) for shard assignment and access patterns


### Short answer

If your Employees table stores **exactly 10 GB**, DynamoDB will typically have **1 physical partition** (because a partition holds ~10 GB).

### Exact rule and calculation (concise)

Partition count = **max( ceil( table_size_bytes / 10GB ), ceil(provisioned_WCU / 1000), ceil(provisioned_RCU / 3000) )**

* For **size**: `ceil(10 GB / 10 GB) = 1`
* If you haven’t provided provisioned throughput, size → **1 partition**.

### Important caveats (must-know)

* AWS uses **≈10 GB per partition** as the storage threshold; the exact split behavior is internal/approximate.
* DynamoDB may allocate extra partitions for operational headroom or throughput even when size alone suggests 1.
* Partitions split **only** when size grows above the threshold (not due to throughput). Throughput-driven needs can still cause adaptive capacity actions and redistribution but not a split based purely on traffic.
* GSIs and Streams have their own partitions and capacity considerations — they increase the effective number of partitions billed/used.

### How to verify in your account (practical)

1. Check table size with AWS CLI:

```bash
aws dynamodb describe-table --table-name Employees --query "Table.{SizeBytes:TableSizeBytes,ItemCount:ItemCount}"
```

2. Inspect provisioned throughput (DescribeTable) and CloudWatch metrics (`ConsumedReadCapacityUnits`, `ConsumedWriteCapacityUnits`) to compute required partitions using the formula above.
3. If you need help computing partitions including your RCU/WCU or GSIs, tell me those numbers and I’ll compute it for you.
### Assumptions and key facts (short)

* You provisioned **RCU = 1500**, **WCU = 500**, and table storage = **10 GB** → **initial partition count = 1** (10 GB ≈ 1 partition; throughput needs also fit into one partition minimum).
* **WCU/RCU cost is item-size dependent**: a 1 KB write typically consumes 1 WCU; larger items consume more (rounded-up per 1 KB).
* Components that act: **DynamoDB control plane (partition manager)**, **Adaptive Capacity engine**, **Partition router / burst-credit subsystem**, and **Application Auto Scaling** (only if configured).

### What happens at **1000 writes/sec** (behavioral summary)

1. **WCU demand vs provisioned**

   * If each write = 1 WCU (1 KB item), demand = **1000 WCU/sec** while provisioned = **500 WCU** → table is under-provisioned.
2. **Immediate outcome**

   * Requests will start to be **throttled** (`ProvisionedThroughputExceededException`) once consumed capacity exceeds provisioned + any available burst credits.
3. **Short-term mitigation by DynamoDB internals**

   * **Burst credits**: partition can absorb short bursts using credits (a few minutes).
   * **Adaptive Capacity**: may temporarily route extra capacity to hot keys and avoid some throttling — but adaptive capacity borrows unused capacity from *other partitions*. You have only **one partition**, so scope for borrowing is minimal.
4. **If Application Auto Scaling is configured**

   * Auto Scaling will detect high utilization and attempt to **raise WCU**, but scaling is not instantaneous (usually tens of seconds to a few minutes). During that window you will see throttling.
5. **If writes are to a single hot key vs many keys**

   * Single hot key → throttling faster and harder (item-level limits matter).
   * Many keys that all hash to the same physical partition still overload that partition → same throttling behavior.
6. **Practical result**

   * Without rapid provisioning (or sharding) you will observe throttling at sustained 1000 writes/sec; short bursts may succeed.

### What happens at **10,000 writes/sec** (behavioral summary)

1. **WCU demand**

   * If 1 KB items: demand = **10,000 WCU/sec**.
2. **Required partitions (if you scaled table to match demand)**

   * Rough estimate: DynamoDB partitions support ~1000 WCU each (published guidance used for rough math).
   * To serve 10,000 WCU you would need **≈ ceil(10000/1000) = 10 partitions** (or equivalent capacity across partitions).
3. **Immediate outcome with current provisioned WCU=500**

   * Severe throttling will occur immediately; burst credits and adaptive capacity cannot absorb a sustained 10k/sec load.
4. **If you had auto scaling and high max capacity**

   * Application Auto Scaling would try to raise table WCU. To get to 10k WCU it will allocate more partitions behind the scenes (partition manager will increase partition count to satisfy throughput). This takes some time and costs increase proportionally.
5. **Partition splitting behavior**

   * Partitions split when DynamoDB decides based on throughput/size; to support 10k sustained WCU DynamoDB will allocate more partitions so the load spreads — but this is a control-plane action, not immediate per-request.
6. **If table remains single-partition (no scaling)**

   * The single partition will drop large fraction of requests → persistent throttling until you reduce load or increase WCU.

### Numeric examples (assume 1 KB item)

* Current provisioned WCU = 500 → can sustain ~500 writes/sec (strongly consistent write model for 1 KB items).
* At **1000 w/s**: needed WCU = 1000 → short bursts may be served; sustained load → throttling until WCU increased to ≥1000.
* At **10000 w/s**: needed WCU = 10000 → you must provision ~10k WCU (→ ~10 partitions) or use on-demand with careful ramping; otherwise heavy throttling.

### Cost and operational implications

* Raising WCU to handle sustained 10k w/s is **expensive** (pay for provisioned capacity).
* On-demand mode can help absorb traffic spikes but is not instantaneous for very sharp surges and costs more per request if traffic is high and sustained.
* Sharding (adding shardId in PK) + increasing provisioned WCU or invoking auto-scaling gives better stability: sharding spreads keys across partitions once DynamoDB has multiple partitions to map hashes to.

### Recommendations to avoid throttling (practical)

* **Enable Auto Scaling** with appropriate min/max WCU and aggressive target utilization (but expect some lag). Component: Application Auto Scaling + DynamoDB control plane.
* **Introduce write sharding** (add shardId in PK) if a small set of keys are hot. This spreads load across partition keys so when partitions increase they distribute load better.
* **Use batching / SQS + worker pool** to smooth spikes (buffer writes).
* **Consider On-Demand mode** if spikes are unpredictable and short-lived — but test large spike behavior.
* **Implement exponential backoff retries** on the client side to handle transient throttling gracefully.
* **Monitor CloudWatch metrics**: Consumed/ProvisionedCapacity, ThrottledRequests, SystemErrors, BurstCreditBalance.

### Final concise summary

* At **1000 w/s** with WCU=500: expect throttling for sustained load; short bursts may succeed using burst credits; auto scaling or manual increase to ~~1000 WCU needed.
* At **10000 w/s** with WCU=500: severe throttling unless you scale WCU to ≈10,000 (which prompts DynamoDB to allocate ~10 partitions) or redesign (sharding + buffering).
* Key levers: **provisioned WCU**, **number of partitions (driven by throughput/size)**, **adaptive capacity**, **burst credits**, and **auto scaling**; these are the components that decide whether requests are served or throttled.


Here is your next **DynamoDB difficult / interview-level question** (AWS Solution Architect Pro level):

---

### ### **Difficult DynamoDB Question**

You have a **DynamoDB table** that stores **IoT sensor events** with the following schema:

* **PK** = `deviceId`
* **SK** = `timestamp`
* **Provisioned**:

  * **WCU = 6000**
  * **RCU = 3000 (eventual)**
* **Item size = 2 KB**
* **Table size = 40 GB**
* **No Auto Scaling**
* **80 devices**, but **2 devices** are extremely hot, sending **1500 events/sec each**
* All devices send data with increasing timestamps.

**Questions:**

1. **How many physical partitions will DynamoDB allocate for this table?**
   (Consider *both* throughput and size-based partitioning rules.)

2. **Will the two hot devices experience throttling?**
   Explain using:

   * Per-partition throughput limits
   * Per-partition hot key limits
   * Adaptive capacity behavior

3. **Without changing provisioned WCU**, what **data model change** would allow both hot devices to sustain 1500 events/sec without throttling?
   Explain why it works **internally** in DynamoDB’s partition router.

4. **If you switch the table to On-Demand**, and the two devices suddenly spike to **5000 writes/sec each**,
   what happens during the **first minute**, and why?
   Explain the behavior of On-Demand warm-up and burst absorption.

---

If you want, answer each part and I’ll grade your response like an AWS Principal Engineer interview reviewer.


### Part 1 — How many physical partitions will DynamoDB allocate?

Compute both throughput- and size-driven partition needs and take the max.

Facts / published guidance used for the calculation (rules-of-thumb used by DynamoDB internals):

* Approx per-partition throughput baseline: **~1000 WCU** and **~3000 RCU** (used for capacity-driven partition allocation).
* Partition storage threshold: **~10 GB per partition**.

Given your inputs:

* Table WCU = **6000**
* Table RCU = **3000 (eventual reads, but we only need RCU for partitions)**
* Table size = **40 GB**

Calculations:

* Partitions for writes = `ceil(6000 / 1000) = 6`
* Partitions for reads = `ceil(3000 / 3000) = 1`
* Partitions for size = `ceil(40 GB / 10 GB) = 4`

Final partition count = `max(6, 1, 4) = 6` → **DynamoDB will allocate 6 physical partitions** (control plane may add operational headroom but 6 is the minimum based on throughput/size rules).

Components involved: DynamoDB **partition manager / control plane** decides partition count and assigns capacity.

---

### Part 2 — Will the two hot devices (1,500 events/sec each) be throttled? (detailed reasoning)

First convert to WCU demand (item size = 2 KB ⇒ each write ≈ 2 WCU):

* Each device: `1,500 writes/sec × 2 WCU = 3,000 WCU`
* Two hot devices combined = `6,000 WCU` → equals total table WCU.

How those writes map to partitions: with PK = `deviceId`, each device’s writes map to a single partition hash value (each hot device hashed to one of the 6 partitions). So per-partition demand for the partitions holding the two hot devices:

* Partition A (hot device 1): **3,000 WCU demand**
* Partition B (hot device 2): **3,000 WCU demand**

What each partition can provide initially:

* Baseline per-partition capacity ≈ **1,000 WCU**
* DynamoDB **adaptive capacity** can boost a hot partition’s usable throughput by borrowing unused capacity or giving per-key boosts (practical boosts often allow up to ~2,000 WCU per hot key in many real-world cases, but AWS does not publish a hard fixed number).
* There are also **burst credits** that can absorb short spikes (minutes).

Practical outcome:

* Because each hot device demands **3,000 WCU** on its single partition, and the typical adaptive boost per partition/hot key is less than 3,000 WCU, **each hot device will experience throttling** unless additional capacity is provided.
* Adaptive capacity can help to some degree by shifting unused capacity from other partitions to the hot partitions, but it has limits and is not guaranteed to satisfy sustained 3,000 WCU for two separate partitions simultaneously. Here the table-wide WCU (6,000) equals demand, but per-partition physical limits and adaptive limits cause contention: adaptive capacity must split the unused capacity across partitions and per-key hard limits can still apply.
* **Burst credits** may absorb part of the demand for a short window (a few minutes) but will exhaust quickly under sustained load.
* Result: With the current schema and provisioned capacity, you should expect throttling on the two hot device partitions under sustained 1,500 events/sec/device.

Internal components enforcing behavior: **partition router**, **partition-level request scheduler**, **adaptive capacity engine**, and **burst-credit subsystem**.

---

### Part 3 — Data-model change (without increasing provisioned WCU) to allow both hot devices to sustain 1,500 events/sec each

Change: **write-shard each hot device** (logical sharding / bucketed partition key). Example:

* New PK design: `PK = deviceId#shardId`
* `shardId` is a small integer you pick per write (e.g., 0..1 for 2 shards, 0..3 for 4 shards). Choose shard assignment by round-robin or hash(timestamp) so writes for the same device are split across `N` logical partition keys.

Why this works (internal DynamoDB reasoning):

1. **Hash diversification:** DynamoDB hashes the partition key to pick a physical partition. `deviceA#0`, `deviceA#1`, ... produce different hash values and thus map to different physical partitions (or to different hash slots that will map among existing partitions).
2. **Per-shard WCU demand drops:** If you use 2 shards per device, each shard receives ~750 writes/sec → 750 × 2 WCU = **1,500 WCU** per shard. That per-shard load is within what adaptive capacity / per-partition baseline can handle much more comfortably.
3. **Parallelism across partitions:** With 6 physical partitions in the table, the sharded keys will distribute across them. Writes for a single logical device no longer concentrate on a single partition; they spread, avoiding hot-partition throttling.
4. **No table-wide WCU increase required** because total WCU demand remains the same (6,000) — you’ve only removed per-partition concentration.

Recommended shard sizing (calculation example):

* Per-device demand = 3,000 WCU. Choose `num_shards = ceil(3,000 / target_per_shard_capacity)`. If you want each shard ≤ 1,500 WCU (safe), `num_shards = ceil(3000 / 1500) = 2`. So **2 shards per hot device** suffice here. If you want even more headroom, use 4 shards → 750 WCU per shard.

Operational notes:

* You’ll need an access pattern to read these sharded items (e.g., query all shards for a device and merge results, or write the consumer so it reads specific shard(s) it needs). For time-ordered reads, store timestamp in SK and do parallel queries for each `deviceId#shardId`.
* For ordering guarantees per device, choose a sharding scheme that preserves intra-shard ordering if required (e.g., round-robin + timestamp allows near-ordered per shard).

Components impacted: **partition router** will now map different `deviceId#shard` keys across many partitions, and **adaptive capacity** will have less pressure per partition.

---

### Part 4 — If you switch to On-Demand and devices spike to 5,000 writes/sec each, what happens in the first minute?

Convert to WCU first (2 KB per item ⇒ 2 WCU each):

* Each device: `5,000 writes/sec × 2 WCU = 10,000 WCU`
* Two devices combined = `20,000 WCU` sustained

On-Demand behavior summary (first minute):

1. **Rapid initial serving with some smoothing:** On-Demand mode dynamically allocates capacity. For sudden spikes, on-demand generally serves a lot more traffic than a small provisioned table could, using internal scaling mechanisms and burst capacity. DynamoDB attempts to serve the spike immediately by allocating more backend resources.
2. **Short transient throttling possible:** On-demand has internal rate-of-growth smoothing. Extremely sharp spikes (like from near-zero to tens of thousands of WCU within one second) can see **brief throttling** while the on-demand engine allocates capacity across partitions and warms up. You may observe some `ProvisionedThroughputExceededException`/HTTP 429 responses in that warm-up window.
3. **Adaptive capacity + on-demand combine:** DynamoDB will use burst credits and adaptive capacity alongside on-demand scaling logic to absorb the initial burst. In practice many on-demand tables will serve sustained tens of thousands of writes/sec after a short ramp, but the short ramp can be noisy.
4. **Cost impact:** On-demand pricing charges per request (higher cost per request compared to provisioned for sustained high throughput). Serving 20k WCU for a minute can be expensive relative to provisioned+autoscaling.
5. **If spikes continue and exceed internal controls:** If the spike is beyond AWS internal soft limits for a single table or region, temporary throttling will persist until control plane adds capacity. For predictable large sustained load, AWS recommends pre-warming / contacting support or using provisioned mode with sharding and autoscaling.

Internal components involved during on-demand warm-up: **on-demand scaling controller**, **partition manager**, **adaptive capacity engine**, and **burst-credit subsystem**.

Practical recommendation for safety: For extremely large instantaneous spikes, combine on-demand with a buffering layer (Amazon Kinesis / SQS) and workers that write at a controlled rate, or pre-provision capacity and enable Application Auto Scaling with a high max.

---

### Concise final recommendations (actionable)

* For sustained hot-device workloads: **shard the deviceId** (2–4 shards per very hot device) so per-shard WCU stays within per-partition comfortable limits.
* Add **buffering** (SQS/Kinesis) and **exponential backoff** for clients to smooth bursts.
* For unpredictable spikes, **On-Demand** helps but test behavior for your spike shape; for extremely large sustained traffic, provision capacity + autoscaling or contact AWS.
* Monitor these CloudWatch metrics closely: `ConsumedWriteCapacityUnits`, `ThrottledRequests`, `BurstCreditBalance`, and per-partition throttling signals.

If you want, I can:

1. produce concrete shard-count math for other write rates,
2. provide sample code for sharding and reading shards in parallel, or
3. simulate CloudWatch metric thresholds and autoscaling policies for this exact workload. Which would you like next?

