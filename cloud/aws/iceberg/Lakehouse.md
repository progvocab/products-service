# Cost-Optimized Data Lakehouse with Iceberg on AWS

Below is a complete, practical design that addresses the 5 requirements (minimize cost, schema evolution, time-travel, small file problem, ACID across partitions). It combines streaming + batch writes, Iceberg table features, S3 cost controls, and operational practices.


---

1) High-level architecture (components)

Ingestion

Near-real-time: Kinesis Data Streams → Kinesis Data Analytics (Apache Flink) or Flink on EMR/EMR on EKS → Iceberg sink (write Parquet/ORC files to S3 via Iceberg API).

Batch: AWS Glue or Spark on EMR → Iceberg write (Glue Spark job using Iceberg libraries).


Storage & Metadata

S3 as the data lake (bucket(s) per environment). Use prefix layout like s3://company-lake/raw/..., s3://company-lake/iceberg/….

Iceberg tables for table format (metadata + snapshot/time-travel). Use Parquet (or ORC) files.

Catalog: AWS Glue Data Catalog (Iceberg can use a Glue catalog or a Hive/ICEBERG catalog backed by Glue). This lets Athena/Glue/EMR discover tables.


Querying

Athena (engine with Iceberg support) for ad-hoc SQL and BI.

Presto/Trino or EMR/Spark for complex analytics.


Orchestration & Jobs

Glue workflows / Step Functions / Managed Airflow for scheduled compactions, metadata maintenance, snapshot expiry.


Security & Governance

SSE-S3 or SSE-KMS for object encryption; bucket policies; IAM roles; optionally Lake Formation for row/column permissions.




---

2) Minimize S3, Athena, Glue costs

Practical tactics:

S3

Use columnar format (Parquet/ORC) to reduce storage and scan costs.

Tune target file size (see below) to avoid many tiny files.

Use S3 Intelligent-Tiering or lifecycle rules: keep recent data in STANDARD, transition older data to INTELLIGENT_TIERING/GLACIER when cold.

Enable S3 server-side encryption with KMS (SSE-KMS) only if required (costs vs SSE-S3).


Athena

Use Iceberg table format — Iceberg metadata avoids full S3 listing for queries and allows pruning.

Partition pruning & partition projection: design partitions that Athena can prune easily (and use Iceberg’s hidden partitioning to avoid many small directories).

Push filters down (Iceberg does this) and minimize full-table scans.

Use workgroup query limits and cost controls.


Glue

Use serverless Glue v3/v4 when jobs are bursty; prefer pre-warmed EMR for high-throughput compaction jobs if more cost-effective.

Reuse connections and avoid large numbers of tiny Glue jobs — consolidate.


Other

Catalog metadata caching (Athena has caching), avoid frequent full metadata refreshes.



---

3) Schema evolution & Time-travel

Iceberg features to use:

Schema evolution: Iceberg supports adding/dropping columns and field renames without rewriting data. Use Iceberg alter table operations (ALTER TABLE ... ADD COLUMN) via Spark/Glue.

Hidden partitioning: Use Iceberg partition transforms (e.g., days(ts) or bucket(...)) so partitioning is logical metadata, not physical directory proliferation. This makes schema evolution and partition evolution easier.

Time-travel: Iceberg snapshots and snapshot IDs allow querying historical state, e.g. SELECT * FROM table FOR SYSTEM_TIME AS OF <snapshot-id/timestamp> (supported by Athena/Trino/Spark).

Retention: Implement snapshot retention policies — e.g., keep snapshots for 30 days then expire to reduce metadata & storage. Use expire_snapshots + remove_orphan_files.



---

4) Avoid small file problem

Design and operational controls:

Writer configuration

Target file size — configure writers (Flink/Spark) to produce files around 256–1024 MB depending on query patterns (common: 512 MB). Iceberg and Parquet support large files that are more cost-efficient for read throughput.

Buffering & micro-batching: In streaming, buffer events for a short window (e.g. 1–5 min) to create larger files rather than writing per event.


Compaction

Continuous/periodic compaction jobs that rewrite small files into large ones:

Use Iceberg's rewriteDataFiles (Spark) or Rewrite Data Files utilities.

Schedule based on incoming volume (e.g., hourly near-real-time compaction for hot partitions, daily for cold).


Partition-aware compaction: compact only partitions with many small files.


Auto-compaction patterns

Small file threshold triggers (e.g., >100 files in partition or average file size < 64 MB).

Use AWS Lambda or Step Functions to trigger Glue/EMR compaction when threshold exceeded.


Metrics & alerts

Collect metrics (number of files per partition, average file size). Trigger compaction if thresholds crossed.



---

5) Ensure ACID transactions across partitions

Iceberg guarantees ACID semantics and snapshot isolation when used correctly:

Use Iceberg table atomic APIs

Always write/commit via Iceberg’s writer/commit API (Flink Iceberg sink, Spark Iceberg write). These produce atomic commits (metadata transaction). Avoid ad-hoc S3 writes directly into table directories.


Concurrent writers

Use Flink or Spark single writer per table partition semantics or rely on Iceberg’s optimistic concurrency handling (it will fail conflicting commits). Implement retry with backoff and compaction on conflict.

For high concurrency, use multi-writer patterns supported by Iceberg/Flint: Flink + Iceberg provides exactly-once semantics when configured with checkpointing.


Transaction workflows

For multi-step ETL that must be atomic across partitions, use Iceberg snapshot + commit operations. Make change in a transaction and commit once all changes are ready.


Validation

After compaction or major writes, run integrity checks (row counts, manifests) and maintain audit logs via Iceberg metadata.



---

6) Implementation details & concrete suggestions

Table design & partitioning

Use Iceberg partition spec with transforms:

Example: partition by days(ts) for time queries or bucket(16, user_id) for even data spread.


Avoid extremely high cardinality partitions (e.g., user_id as partition). Use bucketing or transforms instead.


File format & sizes

Use Parquet with snappy compression for balance of CPU and compression.

Target file size: 256–1024 MB (start with 512 MB). Tune based on query latency and throughput.


Compaction job example (conceptual)

Glue/EMR Spark job:

Read Iceberg table (Spark + Iceberg jars)

table.rewriteDataFiles().option("target-file-size-bytes", 536870912).execute()

Run expire_snapshots(olderThan) and remove_orphan_files() periodically.



Streaming writers

Kinesis Data Analytics (Flink) + Iceberg sink with checkpointing enabled for exactly-once writes.

Configure Flink to write file sizes close to target and use Iceberg commit coordination.


Snapshot retention & housekeeping

Keep snapshots for short retention window for cheap storage and prune older snapshots:

table.expireSnapshots().expireOlderThan(30 days) then removeOrphanFiles.


Keep manifests for a reasonable window if frequent time-travel is required.


Metadata & Catalog

Use AWS Glue Data Catalog as Iceberg catalog. Register Iceberg tables there so Athena and Glue can query them.

Keep catalog and metadata backups if needed.



---

7) Operational runbook & monitoring

Metrics to collect:

average file size per partition

number of files per partition

snapshot count & metadata size

Glue/Athena query cost & scan bytes


Alerts:

Files per partition > threshold

Average file size < threshold


Jobs:

Hourly/near-real-time compaction for hot partitions

Daily/weekly major compaction for cold partitions

Snapshot expiration daily


Testing:

End-to-end test for schema evolution (add/remove columns)

Time-travel queries to previous snapshot IDs




---

8) Example tradeoffs / tuning knobs

Aggressive compaction → fewer files, lower Athena cost, but higher Glue/EMR job cost.

Larger files → cheaper reads but increased latency for near-real-time consumers who need latest rows.

Long snapshot retention → better time-travel but more metadata storage and higher S3 costs. Tune based on query patterns and cost targets.



---

9) Summary checklist (practical)

1. Choose Parquet + Iceberg; use Glue catalog.


2. Streaming: Flink (KDA) → Iceberg sink; Batch: Glue/EMR Spark → Iceberg.


3. Configure writers to target ~512MB files; enable buffering/micro-batching.


4. Use Iceberg hidden partitioning and bucketing for even distribution.


5. Run scheduled compaction (rewriteDataFiles) & housekeeping (expireSnapshots, removeOrphanFiles).


6. Apply S3 lifecycle & Athena partition pruning; monitor file metrics and query scan bytes.


7. Use Iceberg snapshot features for time-travel; set retention policy.


8. Secure with SSE-KMS, bucket policies, and IAM roles (or Lake Formation).




---

If you want, I can:

Draft sample Spark/Flink config snippets for Iceberg writes (producer/compaction examples).

Provide a cost comparison table (compaction frequency vs Athena scan cost) based on your query profile.

Ask follow-up questions about your throughput (TB/day), latency SLA for near-real-time workloads, and preferred query engines — then give tuned numeric recommendations.

