ClickHouse offers a variety of **table engines** to support different use cases including OLAP queries, data ingestion, real-time analytics, and external data access. Here’s a categorized overview:

---

### **1. MergeTree Family (Used for OLAP and large-scale data storage)**

These engines store data **in parts**, allow for background **merging**, and are designed for **fast analytical queries**:

| Engine               | Description |
|---------------------|-------------|
| `MergeTree`         | Base engine, supports indexing and partitioning. No deduplication. |
| `ReplacingMergeTree`| Removes duplicates based on a version column or key. |
| `SummingMergeTree`  | Aggregates rows with the same key during merges. |
| `AggregatingMergeTree` | Stores pre-aggregated data (via `aggregateFunction`). |
| `CollapsingMergeTree`| Supports row-level deletion using `sign` column. |
| `VersionedCollapsingMergeTree` | Like `CollapsingMergeTree`, but supports versions. |
| `GraphiteMergeTree` | For Graphite-style time-series data (deprecated for most use cases). |

---

### **2. Log Engines (Simpler, faster for insert-heavy workloads, no background merging)**

| Engine      | Description |
|-------------|-------------|
| `Log`       | Simple log, appends data as-is. Suitable for tiny datasets. |
| `StripeLog` | Columnar variant of `Log`. |
| `TinyLog`   | Stores each column in a separate file; not recommended for production. |
| `Memory`    | Stores data entirely in memory. Data is lost on restart. Useful for temp tables. |

---

### **3. Buffer and Kafka Engines (For streaming and real-time ingestion)**

| Engine       | Description |
|--------------|-------------|
| `Buffer`     | Writes to memory and flushes to another table periodically or by size. |
| `Kafka`      | Connects to Apache Kafka, acts as a source table for streaming ingestion. |

---

### **4. External Table Engines (Query data outside ClickHouse)**

| Engine        | Description |
|---------------|-------------|
| `JDBC`        | Access external databases via JDBC. |
| `ODBC`        | Same as JDBC, but using ODBC drivers. |
| `MySQL`       | Live queries to MySQL tables. |
| `PostgreSQL`  | Live queries to PostgreSQL. |
| `S3`          | Query Parquet/CSV/JSON data stored in Amazon S3 or compatible storage. |
| `HDFS`        | Read data from Hadoop HDFS. |
| `URL`         | Fetch and read data from HTTP(s) endpoints. |

---

### **5. Join Engines**

| Engine     | Description |
|------------|-------------|
| `Join`     | Used to preload join tables into memory for faster lookups. |
| `Dictionary` | Used with external dictionaries (not a table engine per se). |

---

### **6. Special-Purpose Engines**

| Engine         | Description |
|----------------|-------------|
| `Merge`        | Virtual table that merges data from other tables (of same structure). |
| `Distributed`  | Queries across multiple shards in a cluster. |
| `Null`         | Discards inserted data; always returns empty results. |
| `File`         | Reads from local files (CSV, TSV, etc.) — temporary usage. |
| `MaterializedView` | Automatically stores query results from another table or query. |

---

Would you like a visual table of when to use which engine based on use case?