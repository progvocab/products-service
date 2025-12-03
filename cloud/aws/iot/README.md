Nice — let’s build this. I’ll answer the full design end-to-end and include capacity arithmetic, tradeoffs and mitigations.


---

Summary (one-line)

Use edge batching + AWS IoT Core → high-throughput streaming (MSK/Kinesis) → real-time in-memory window (MemoryDB/Redis cluster) + query layer + Firehose → S3 cold store. Enforce per-tenant isolation with per-tenant ingress tokens, per-tenant shards/partitions or throttles, and separate memory/compute reservations for heavy tenants. Meter everything into a billing pipeline.


---

1) Workload numbers (careful arithmetic)

We must size the pipeline from the device rate.

User gave: 5,000,000 devices sending 2–5 events/sec. I’ll assume a middle value for planning: 3 events/sec average.

Calculate events/sec:

5,000,000 × 3 = 15,000,000 events/sec.


Data per event — pick a conservative 1 KB payload (JSON + metadata).
Per-second bytes:

15,000,000 events/sec × 1,024 bytes/event = 15,360,000,000 bytes/sec.


Convert to GB/s:

15,360,000,000 bytes/sec ÷ 1,024 = 15,000,000 KB/sec.

15,000,000 KB/sec ÷ 1,024 = 14,648.4375 MB/sec.

14,648.4375 MB/sec ÷ 1,024 ≈ 14.3125 GB/sec. (≈14.3 GB/s)


Per hour:

14.3125 GB/sec × 3600 sec = 51,525 GB/hr ≈ 50.3 TB/hr (decimal vs binary rounding differences aside).


Per day:

50.3 TB/hr × 24 ≈ 1,207 TB/day ≈ 1.18 PB/day.


Takeaway: raw numbers are enormous. Edge aggregation, compression, batching, smaller payloads, sampling and client-side filtering are essential to make this practical/cost-effective.


---

2) High level architecture (components)

Devices → Edge (optional: Greengrass / local gateway) → AWS IoT Core (regional, multi-AZ) → Stream ingestion layer (Amazon MSK / self-managed Kafka on EKS or high-shard Kinesis Data Streams) → Real-time processing (Kinesis / Kafka consumers: Flink or Kinesis Data Analytics / ksqlDB) → Real-time store (MemoryDB for Redis / ElastiCache Cluster Mode) for last-1-hour windows → API layer / Query service (auto-scaled ECS/Fargate or Lambda fronted by API Gateway) → Firehose → S3 (partitioned by tenant/date) for cold/historical.
Billing & metrics pipeline reads stream of metering events and aggregates per tenant.

(Each component is deployed multi-AZ or regional for resiliency.)


---

3) Data ingestion design

Goals: high throughput, low latency, per-tenant throttling, durability.

Flow and choices:

1. Device/Edge

Prefer edge gateways (AWS IoT Greengrass or simple local gateways) that batch and compress events, apply client-side rate limiting, and do local dedup. This reduces load on the cloud and provides resilience for short network interruptions.

Use MQTT or HTTP with TLS. Each device authenticates with unique cert or token.



2. AWS IoT Core

Handles secure connection, authentication, TLS, MQTT retain/last-will semantics. IoT Core is regional and resilient across AZs.

IoT Rules route messages to Kafka/MSK/Kinesis Firehose/Lambda. Use filtering to drop malformed messages early.



3. Stream layer options (choose based on throughput and ops preferences)

Amazon MSK (managed Kafka) — better throughput per partition/replica than Kinesis shards; good for extremely high event rates (15M r/s). MSK supports cross-AZ replication; you can configure partitions per tenant or per tenant group.

Kinesis Data Streams — simple serverless model but each shard gives 1 MB/s or 1000 rec/sec. For 15M rec/sec you need far too many shards, so Kinesis would be expensive unless you batch heavily and reduce records/sec. Kinesis Data Firehose is excellent for streaming to S3 with buffering/compression for cold store.




Recommendation: IoT Core → MSK (or self-managed Kafka clusters on EC2/EKS with MSK preferred for managed ops), then run Flink/kSQL consumers for real-time processing, and Firehose for S3.

4. Durability & ordering

Partition data by tenant_id (hash) + time slot so each tenant’s events map deterministically to a partition. For hot tenants, allocate extra partitions.



5. Backpressure and buffering

Devices/gateways should buffer on network issues. On server side, keep the streaming cluster autoscaled and use tiered ingress controls.





---

4) Per-tenant throttling & isolation

Strict isolation tech + operational controls:

1. Ingress authentication & quota

Issue per-tenant credentials (IoT certs or API keys). Use AWS IoT Core policies or API Gateway usage plans to limit RPS per tenant at the edge of the system. Enforce token bucket on gateway level.



2. Partitioning

Map tenants to dedicated partitions or partition groups in MSK/Kinesis. This prevents a noisy tenant from monopolizing a partition shared across tenants.



3. Per-tenant rate tokens in the ingest tier

Implement per-tenant token buckets in a distributed metering service (e.g., Redis or DynamoDB). When capacity is exhausted, either queue or reject with a rate limit response.



4. Per-tenant compute reservation

For real-time compute (Flink tasks, Redis shards), reserve capacity per tenant. For MemoryDB/ElastiCache, use logical namespaces and separate clusters for large tenants.



5. Traffic shaping and circuit breakers

Automatic throttling if tenant exceeds a threshold; escalate to manual review if sustained.



6. Billing-driven isolation

Map tiers: basic tenants share pooled resources with strict quotas; premium tenants get reserved partitions/memory/throughput.





---

5) Real-time query layer (requirement: <2s latency for last 1 hour)

Design pattern: materialized sliding window in memory + optional precomputed aggregates.

1. Real-time state store

Use MemoryDB for Redis (or ElastiCache Cluster Mode) as the sliding-window store. Keep last 1 hour per tenant, either as time-bucketed lists or sorted sets. Redis lets you provide <10 ms reads.

For 5M tenants you cannot allocate per-tenant keys in single cluster without planning; store tenants in a sharded Redis cluster and reserve extra nodes for hot tenants.



2. Ingest → Aggregation

Stream consumers (Flink or Kinesis Data Analytics) consume MSK/Kinesis and update aggregates/windows in Redis in near real-time. Also compute rollups (per-sec, per-10s, per-minute).



3. Query API

API Gateway → ECS/Fargate (or Lambda for simpler) services that read Redis and return results. For heavy queries, add caching layer (CloudFront or API layer caching).



4. Handling large queries

For queries across large time ranges or full history, route to long-term store (Athena/Redshift/Snowflake) and accept higher latency.



5. Scaling for latency

Redis cluster modes, MemoryDB multi-AZ, and auto-scale compute for consumers. Precompute or approximate aggregates (sketches) for very large tenants.





---

6) Long-term storage and analytics

1. S3 (partitioned) — Firehose writes compressed, partitioned Parquet/ORC files to S3: s3://bucket/tenant_id=.../year=.../month=.../day=.../hour=.... Use Parquet and compression for cost savings and faster query.


2. Catalog & query

Maintain AWS Glue catalog and query historical data with Athena or Redshift Spectrum. For heavy OLAP, use Amazon Redshift or an analytical DB like ClickHouse (self-managed) or Snowflake.



3. Cold archiving

Lifecycle rules: S3 Standard → S3 IA after 30 days → S3 Glacier Deep Archive after 180 days.



4. Index for searching

For text/search on recent data, maintain an OpenSearch cluster for the last N days/weeks.





---

7) AZ outage and high-availability plan (≤5s ingestion delay)

Requirements: continue ingest, no data loss, ≤5s ingestion delay.

1. Regional architecture

Deploy MSK, IoT Core, MemoryDB, and consumers across multiple AZs in the same region — AWS ensures cross-AZ replication for MSK and MemoryDB.



2. Device-side buffering

Devices and edge gateways must buffer locally for up to at least 30s–minutes. If an AZ fails, remaining AZ brokers accept connections and regional endpoints route automatically.



3. MSK / Kafka replication

Use replication factor 3 across AZs; producers will failover to remaining leaders. Replication ensures no data loss if full AZ fails.



4. Stateless consumer scaling & leader election

Consumers are in an auto-scalable group across AZs; use Kafka consumer group rebalancing.



5. MemoryDB

Use MemoryDB Multi-AZ with automatic failover. For Redis, use cluster mode with replicas.



6. S3 durability

S3 is regional and multi-AZ by design.



7. Targeting ≤5s ingestion delay

With regional brokers and device buffering, failover should be under seconds. Ensure small producer timeouts and quick reconnection logic on clients; devices should retry with exponential backoff but keep some buffer. Use health checks and alerting to detect failovers.





---

8) Tenant-level billing (metering and attribution)

1. Metering events

At ingestion, every event should include tenant_id. Emit metering events to a billing stream (separate Kafka topic) containing counts, raw bytes, and resource usage markers (e.g., special events when a compute job runs on tenant data).



2. Aggregate pipeline

Use Flink/consumer to aggregate per-tenant metrics: events/sec, bytes, storage bytes (S3 object size), compute seconds (ECS/EMR/Redshift queries measured via logs/cloudwatch), Redis memory usage. Store aggregations in a billing DB (DynamoDB/Redshift/TimeSeries DB).



3. Attribution of infra cost

Map infra costs to tenants using recorded usage. For shared resources, allocate proportionally by measured usage (e.g., total Redis memory cost proportionally split by per-tenant memory consumption).



4. Invoicing & integration

Daily/monthly invoices generated from billing DB. Offer real-time dashboards (Quicksight or custom UI) with current usage and alerts near thresholds.



5. AWS cost tools

Use cost allocation tags where possible and AWS Cost and Usage Reports for infra costs to reconcile.





---

9) Cost optimization strategies

Edge batching & compression — reduces request count and bytes.

Use columnar formats (Parquet/ORC) + compression for S3 to reduce storage and Athena scan costs.

Lifecycle policies to move older data to infrequent access and Glacier.

Tiered tenancy — shared pool for small tenants, reserved resources for big tenants. Charge heavy tenants more.

Spot instances for batch analytics and use reserved/ savings plans for stable baseline.

Avoid over-sharding — combine tenants when low-throughput and only isolate hot tenants.

Use serverless where predictable and cheaper (e.g., Firehose + Athena for ad-hoc analytics) but replace with provisioned infra for sustained high throughput.



---

10) Potential bottlenecks and mitigations

1. Ingress throughput

Bottleneck: producers + broker capacity.

Mitigation: edge batching; MSK scaled partitions; pre-sharding per tenant; use larger instances for brokers; use compression.



2. Hot tenants (noisy neighbor)

Bottleneck: single partition or Redis shard overwhelmed.

Mitigation: detect hot tenants, allocate dedicated partitions/Redis shards, apply per-tenant quotas and circuit breakers, require premium tier for higher throughput.



3. Real-time state size (Redis memory)

Bottleneck: keeping 1 hour of raw events per tenant in memory.

Mitigation: store aggregates rather than raw events, use compressed time buckets, evict less-active tenants, use faster disk-based time-series stores for some tenants (Timestream/ClickHouse).



4. Consumer lag & processing

Bottleneck: consumer group can't keep up.

Mitigation: horizontally scale consumers, use more partitions, tune parallelism (Flink parallelism), monitor consumer lag.



5. Network / AZ failure

Bottleneck: routing and reconnection delays.

Mitigation: multi-AZ replication, device reconnection logic, regional endpoints.



6. Cost explosion

Bottleneck: unanticipated storage or compute use.

Mitigation: enforce quotas, set alerts and automatic throttling, implement cost guardrails per tenant.





---

11) Security and compliance

TLS for device → IoT Core. Mutual auth via device certs.

IAM least privilege for services, KMS for encryption of streams and S3 objects.

VPC for broker and compute; secure endpoints for S3 (VPC endpoints).

Audit logs (CloudTrail), event logging, SIEM integration.

Per-tenant encryption keys if required.



---

12) Observability & SLOs

Track metrics per tenant: ingest r/s, bytes, Redis memory, query latency.

Centralized tracing (X-Ray), logs (CloudWatch/ELK), consumer lag metrics, Redis slowlog.

Alerting: consumer lag thresholds, high retry rates, exceeded quotas.

SLO example: 99.9% of real-time queries <2s; ingestion availability 99.95%; failover ingestion delay ≤5s.



---

13) Example deployment plan & choices

Minimum viable approach (lower ops):

Devices → IoT Core → Kinesis Firehose (buffered, compress to Parquet) → S3 → Athena for analytics.

For real-time: IoT Core → Lambda → DynamoDB (per-tenant recent window). This is easier to run but does not scale well to 15M r/s without heavy batching.


High-scale production approach:

Devices → Greengrass (batch) → IoT Core → Amazon MSK (Kafka) → Flink stream processing → MemoryDB (real-time window) + Firehose→S3 (cold). Use MSK for partitioning and throughput.



---

14) Tradeoffs & final recommendations

If you truly have ~15M events/sec, edge aggregation is mandatory; otherwise cloud costs and complexity explode. Aim to reduce records/sec by batching and compressing at the edge by >10× if possible.

MSK + Flink + MemoryDB gives the best throughput and deterministic latency but requires more ops effort vs serverless Kinesis/Lambda.

Use per-tenant partitions only for truly heavy tenants; for small tenants, use shared partitions with strict quotas.

Prioritize metering from day one — billing accuracy depends on it.



---

15) Short checklist to implement immediately

1. Add tenant_id to every event and produce a metering topic.


2. Implement edge batching (Greengrass or SDK-level batching).


3. Put IoT Core + MSK as ingest backbone.


4. Deploy Flink consumers to compute sliding windows into MemoryDB.


5. Send compressed Parquet files to S3 via Firehose.


6. Start a billing aggregator consumer and validate cost attribution.


7. Add per-tenant rate limiter in API/Gateway or at IoT Core rules.


8. Set monitoring and alerts (consumer lag, hot tenants, Redis memory).




---

If you want, I can now:

Evaluate tradeoffs between MSK vs Kinesis with cost examples (I’ll compute rough monthly cost estimates given a chosen event size and batching level), or

Produce a component diagram + Terraform/CloudFormation skeleton for this architecture, or

Provide a per-tenant throttling algorithm (token bucket implementation) you can drop into the ingestion gateway.


Which of those next steps would be most useful?