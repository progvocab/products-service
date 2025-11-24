Here is a difficult AWS DynamoDB question suitable for a Solution Architect level:


---

DynamoDB Difficult Question

You are designing a multi-tenant SaaS platform that stores customer events in a single DynamoDB table.
Each tenant generates high write throughput, but some "noisy neighbor" tenants occasionally cause hot partitions, impacting overall performance.

You must ensure:

1. Strict tenant isolation of performance


2. Strongly consistent reads for a subset of items


3. Cost-optimized scaling across varying tenant workloads


4. No hot partitions, even if one tenant suddenly spikes 100x in writes



Question:
Explain which partition key design, table design pattern, and DynamoDB feature set (e.g., adaptive capacity, GSIs, on-demand vs. provisioned, token buckets, etc.) you would use to guarantee performance isolation and eliminate hot partitions, while still optimizing cost. Also describe how you would monitor and detect hot partitions at scale.


---

If you want, I can also give: ✔ the answer
✔ next question
✔ even harder SA Pro level questions


Great — here’s a complete, exam-quality answer you can use for an AWS Solutions Architect level response. I’ll cover design, why it works, trade-offs, and how to monitor & remediate hot partitions.


---

Short summary (one-line)

Use a hybrid strategy: put most tenants into a single multi-tenant table with synthetic sharding of the partition key + on-demand/provisioned hybrid capacity and buffering; place very heavy tenants into dedicated tables (or dedicated capacity) for strict isolation; use adaptive capacity, autoscaling, caching, and a write buffer (SQS/Kinesis + worker); monitor using CloudWatch, Contributor Insights, DynamoDB Streams and automated alarms to detect and mitigate hot keys.


---

1) Partition-key / item key design (prevent hot partitions)

Goal: avoid any single partition key receiving > 3–10k RCUs/WCUs for long periods (DynamoDB backend capacity is per partition).

Recommended designs:

A. Synthetic sharding (general multi-tenant table)

PartitionKey = tenantId#shardId

SortKey = eventTimestamp (or eventType#eventTimestamp if you need event type queries)

shardId is a small suffix created by hashing the tenant id and applying mod N or using a random range 0..N-1. Example: tenant42#03.

Choose initial N per tenant by expected peak rate (e.g., start N=10 for medium tenants). For time-series heavy writes combine with time: tenantId#YYYYMMDD#shardId to naturally disperse writes per day.


Why: increasing effective partition key cardinality spreads writes across multiple physical partitions. If a tenant spikes 100x, application can increase N for that tenant.

B. Dedicated tables for "noisy" tenants

If a tenant’s baseline or spike rate is very high or unpredictable, give them a separate DynamoDB table (or dedicated provisioned capacity / reserved capacity). This guarantees tenant-level performance isolation (separate partitions, separate throughput billing).


Trade-off: more management and potentially higher cost, but simple, strongest isolation.


---

2) Table design patterns & features to use

Primary table (multi-tenant):

PartitionKey: tenantId#shardId

SortKey: eventTimestamp (or composite eventType#eventTimestamp)

Sparse GSIs for alternate query patterns (be careful — GSIs have separate partitions & may become hot; use sparse GSIs or sharding on GSI keys too).


Feature choices:

1. On-demand capacity for unpredictable workloads or early stages — good to avoid throttling while you tune sharding. But on-demand can be costlier at scale.


2. Provisioned throughput + Application Auto Scaling (target a utilization like 50–70%) for steady, predictable workloads — cheaper at scale.


3. Adaptive Capacity — DynamoDB will reallocate unused capacity to hot partitions automatically; helpful but not sufficient for sudden, extreme spikes or single huge hot keys.


4. DynamoDB Streams + Lambda workers — for async processing, fan-out, or rebuilding denormalized views.


5. DAX / ElastiCache (read cache) — offload reads from DB to cache to reduce RCU pressure; DAX supports eventually consistent reads only.


6. Transactional writes (TransactWriteItems) only if you need cross-item atomicity — use sparingly due to higher cost/latency.


7. Time-windowed tables (if append-only timeseries) — for extreme ingest volumes, consider rolling tables per month/day and TTL to manage size.


8. Use sparse GSIs or GSIs that include shardId to avoid single GSI hot keys.




---

3) Strongly consistent reads requirement

DynamoDB supports strongly consistent reads only against the base table (single region). Reads against GSIs are eventually consistent.

Design implication: put items that require strong consistency so they can be looked up by primary partition key (or by a query on partition key + sort key that you can perform strongly consistent on). If you must query by alternate attributes and require strong consistency, keep a copy (or pointer) in the main table keyed for that access pattern.

If you need global strong consistency across regions, DynamoDB does not provide cross-region strong consistency — Global Tables are eventually consistent across regions.



---

4) Handling a 100× write spike (no hot partitions)

Apply layered controls:

1. Synthetic sharding: increase N for the tenant (application/server logic chooses larger shardId space) so new writes use more partition keys. Old keys still exist, new writes land on different partitions.


2. Write buffering: put incoming events into SQS/Kinesis Data Streams. Consumers (workers) read from the buffer and write to DynamoDB at a controlled rate. This absorbs spikes without throttling upstream clients and enables backpressure/ retry policies.


3. Token bucket / rate limiter: on the application side or at gateway level to limit write rates per tenant if needed. Works well with a burst allowance.


4. Dedicated table / dedicated provisioned capacity: for large tenants move to their own table to isolate.


5. Adaptive capacity + on-demand: adaptive helps small/medium will rebalance. For unexpected spikes you can use on-demand (if single table) so writes succeed while you reconfigure sharding; but after surge, move to provisioned + sharding for cost efficiency.


6. Backfill and re-shard migration: if you need to change shard count for historical items you may migrate data using a worker reading the old keyspace and re-writing items with new shard suffix. Prefer incremental migration; avoid blocking writes.




---

5) GSI considerations

GSIs have their own partitions and throughput and can become hot. Don’t use tenantId alone as GSI partition key for heavy workloads. Use tenantId#shardId also for GSIs where necessary.

Consider making GSIs sparse (only index items that actually need that access pattern) to reduce index size and hotkeys.

GSIs do not support strongly consistent reads.



---

6) Capacity mode selection & cost optimization

Mixed approach:

Use on-demand for unpredictable or small tenants during rapid growth (no capacity planning).

Use provisioned + autoscaling for predictable, steady workloads (cheaper at scale) with target utilization (e.g., 50%).

For very large/steady tenants: dedicated tables with provisioned capacity, possibly reserved capacity/commitments.


Reserved capacity / Savings Plans if you have steady, known large throughput.

Use DAX or ElastiCache to cut read costs.

Use TTL to prune old events automatically where appropriate.



---

7) Monitoring & detection (how to detect hot partitions at scale)

Metrics to collect (CloudWatch & DynamoDB):

ThrottledRequests (overall and per operation)

ConsumedReadCapacityUnits / ConsumedWriteCapacityUnits

ReadThrottleEvents, WriteThrottleEvents

SuccessfulRequestLatency (P50/P90/P99)

SystemErrors, UserErrors

CloudWatch Contributor Insights for DynamoDB — can identify top partition keys / top throttled keys over time.

DynamoDB Streams + custom Lambda aggregator to compute per-partition counts if you need more granular telemetry.


Concrete detection recipe:

1. Enable Contributor Insights for key patterns (e.g., partition keys like tenantId#shardId) — it will show “top N” keys by write count, throttles, etc.


2. Create CloudWatch alarms:

WriteThrottleEvents > 0 (or > threshold) → trigger investigation/auto-remediation.

ConsumedWriteCapacityUnits per table rising beyond X% of provisioned capacity.

P99 write latency rising above SLO.



3. Collect top partition keys using Contributor Insights or a periodic job that queries CloudWatch metrics for PartitionKey dimensions (or parse Streams events to count writes per key).


4. Visualize (Grafana/CloudWatch dashboards): per-tenant consumed WCUs and throttles.



Automated remediation suggestions (on alarm):

If a single tenant key is hot: temporarily route traffic to buffering (SQS/Kinesis) and increase shard count for that tenant (application starts writing to new tenantId#shardId values). An orchestrator Lambda can update configuration in a central metadata store (DynamoDB "tenant config" item listing current shard count).

If multiple keys hot: raise provisioned capacity (if provisioned), or switch to on-demand for short term.

If a specific GSI hot: consider changing GSI design (add shard suffix), or move that access pattern to a separate table.



---

8) Example implementation plan (step-by-step)

1. Classify tenants: small, medium, large by expected QPS/WPS. Small/medium -> multi-tenant table; large -> dedicated table.


2. Implement sharded partition key: PK = tenantId#shardId, SK = eventTimestamp.


3. Write path:

Clients → API Gateway/LB → Producer service → SQS/Kinesis (optional but recommended)

Consumer workers read from queue and write to DynamoDB using PK.

Workers compute shardId = hash(tenantId + eventId) % N or use round-robin based on tenant config.



4. Read path:

For strongly consistent lookups use GetItem or Query with ConsistentRead=true on the base table (only possible for PK queries).

For complex queries use GSIs (expect eventual consistency) or materialized views kept via Streams.



5. Autoscaling & capacity:

Configure autoscaling policies for provisioned tables; keep safety margins.

For unpredictable spikes, allow automatic fallback to on-demand.



6. Monitoring & alerts: Contributor Insights, CloudWatch alarms on throttles, dashboards, and an auto-remediate Lambda.




---

9) Example partition key formats & sharding logic

Simple fixed shards:

shards = 16 → shardId = hash(tenantId) % 16

PK example: tenant123#05


Dynamic/time based:

PK = tenant123#20251124#07 (useful for very high volume per day; reduces hotness during the day)


Application pseudo:


def pick_shard(tenant_id, event_id, shard_count):
    return hash(tenant_id + event_id) % shard_count


---

10) Pros & cons summary

Pros

Synthetic sharding disperses load and prevents single hot partition.

Buffers (SQS/Kinesis) decouple spikes from DB and provide durability.

Dedicated tables provide absolute isolation for noisy tenants.

Contributor Insights gives visibility into hot keys.


Cons / things to manage

Application complexity increases (shard management + routing + migrations).

Rewriting/migrating existing data to new shard counts is operational work.

GSIs and secondary indexes still need sharding or can become hot.

On-demand is expensive for sustained high throughput.



---

11) Quick checklist you can use in an interview answer

[x] Use tenantId#shardId to spread writes.

[x] Use buffering (SQS/Kinesis) + consumers to smooth spikes.

[x] Put extremely heavy tenants on dedicated tables/capacity.

[x] Use Adaptive Capacity + autoscaling; consider on-demand fallback.

[x] Ensure items requiring strong consistency are read from base table (use ConsistentRead=true).

[x] Monitor with CloudWatch + Contributor Insights + Streams; alert on throttles and top keys.

[x] Automate shard scaling (increase N) and migrate/backfill as needed.

[x] Cache reads (DAX) and use TTL for retention to save cost.



---

If you’d like, I can:

Provide sample code (producer and consumer) showing shard calculation and queueing.

Draft a cloud architecture diagram and a set of CloudWatch alarms with thresholds for your SLOs.

Produce an exam/whiteboard answer condensed to bullets you can speak in 90 seconds.


Which one should I do next?