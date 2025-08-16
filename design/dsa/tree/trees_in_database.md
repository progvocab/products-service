In addition to **B-trees** (and **B+ Trees**), databases use a variety of data structures for indexing depending on the access patterns, data types, and query requirements. Here are the major ones:

---

### 1. **B+ Trees**
- **Used in**: MySQL (InnoDB), PostgreSQL
- **Description**: Variant of B-tree where all values are stored in the leaf nodes and internal nodes store keys for navigation.
- **Best for**: Range queries, sorted data access

---

### 2. **Hash Indexes**
- **Used in**: Redis, PostgreSQL (limited), DynamoDB (internal)
- **Description**: Uses a hash function to map keys to index entries.
- **Best for**: Equality lookups (`WHERE id = 42`), not good for range queries

---

### 3. **GiST (Generalized Search Tree)**
- **Used in**: PostgreSQL
- **Description**: Framework for building custom indexing for various data types.
- **Best for**: Full-text search, geometric data, arrays

---

### 4. **GIN (Generalized Inverted Index)**
- **Used in**: PostgreSQL
- **Description**: Inverted index mapping values to the rows they appear in.
- **Best for**: Full-text search, JSONB, arrays

---

### 5. **R-Trees**
- **Used in**: Spatial databases like PostGIS, SQLite (with R-Tree module)
- **Description**: Hierarchical data structure for indexing multidimensional information like geographical coordinates.
- **Best for**: Geospatial queries (e.g., bounding box)

---

### 6. **LSM Trees (Log-Structured Merge Trees)**
- **Used in**: Cassandra, RocksDB, LevelDB, ScyllaDB, ClickHouse
- **Description**: Writes go to memory first, periodically flushed to disk in sorted order.
- **Best for**: Write-heavy workloads, time series data

---

### 7. **Trie (Prefix Tree)**
- **Used in**: Some full-text engines and memory-optimized key-value stores
- **Description**: Tree structure where common prefixes are shared.
- **Best for**: Auto-completion, prefix search

---

### 8. **Bitmaps / Bitmap Index**
- **Used in**: Oracle, PostgreSQL (with extensions), Druid
- **Description**: Uses bit arrays to represent existence of values.
- **Best for**: Columns with low cardinality (e.g., gender, boolean fields)

---

### 9. **Skip Lists**
- **Used in**: Redis
- **Description**: Layered linked list that allows fast search and insertion.
- **Best for**: In-memory databases needing sorted access with high concurrency

---

Would you like GitHub source code references for any of these implementations?