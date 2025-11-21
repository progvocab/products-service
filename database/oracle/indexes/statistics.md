### Index Statistics in Oracle Database (Concise and Technical)

### Overview

Index statistics in Oracle are metadata collected by the **Cost-Based Optimizer (CBO)** to estimate how efficiently an index can be used during query execution. These statistics directly affect execution plans, index range scans, skip scans, index fast full scans, and the decision to use an index vs. a full table scan. CBO uses these statistics from the **Data Dictionary (USER/DBA/ALL_INDEXES, IND_COLUMNS, IND_STATISTICS)** to compute cardinality, selectivity, and cost.

### Key Index Statistics Tracked by Oracle CBO

### BLEVEL

The number of branch levels in the B-tree (height minus 1).
Lower BLEVEL means fewer I/O operations during an index lookup.

Example:

* BLEVEL = 1 → shallow index → efficient range scans
* BLEVEL = 3 → deeper index → more logical I/O

CBO uses BLEVEL to estimate INDEX RANGE SCAN cost.

### LEAF_BLOCKS

Total number of leaf blocks in the index segment.
Represents the size of key-value storage.

CBO uses this to calculate:

* Full index scan I/Os
* Clustering factor influence for range scans

### DISTINCT_KEYS

Number of distinct indexed key values.
Used by CBO to calculate selectivity = 1 / DISTINCT_KEYS.

High DISTINCT_KEYS → index is more selective.

### CLUSTERING_FACTOR

Measures how well the data in the table is ordered relative to the index.

* Low clustering factor → data is physically ordered; index is efficient
* High clustering factor → table rows are scattered; range scans are expensive

CBO uses this to estimate table block visits during range scans.

### INDEX_TYPE

Type (NORMAL, BITMAP, FUNCTION-BASED, IOT, REVERSE KEY).
CBO chooses algorithms based on index type:

* BITMAP → good for low-cardinality columns
* NORMAL → standard B-tree behavior
* REVERSE → unsuitable for range scans

### NUM_ROWS (From Table Stats but used for Index Costing)

Total rows in the table.
CBO combines table stats with index stats to compute cost.

### LAST_ANALYZED

Timestamp when index stats were last gathered.
Outdated stats may mislead CBO.

### COLUMN_STATISTICS (for index prefix)

Include density, null count, histogram type.
Used by CBO to decide whether to use a prefix or skip scan.

### How to View Index Statistics

```sql
SELECT index_name, blevel, leaf_blocks, distinct_keys, clustering_factor, num_rows
FROM user_indexes
WHERE index_name = 'IDX_ORDERS_CUST';
```

### How to Gather Index Statistics

```sql
BEGIN
  DBMS_STATS.GATHER_INDEX_STATS(
    ownname => 'HR',
    indname => 'IDX_ORDERS_CUST',
    estimate_percent => DBMS_STATS.AUTO_SAMPLE_SIZE,
    method_opt => 'FOR ALL COLUMNS SIZE AUTO'
  );
END;
/
```

### How CBO Uses Index Statistics (Execution Plan Example)

Query:

```sql
SELECT * FROM orders WHERE customer_id = 10;
```

CBO checks:

* DISTINCT_KEYS → selectivity
* BLEVEL → depth cost
* CLUSTERING_FACTOR → estimated table block reads
* LEAF_BLOCKS → full index scan cost

If selectivity is low and clustering_factor is high → CBO may choose FULL TABLE SCAN instead of INDEX RANGE SCAN.

### When to Refresh Index Statistics

* Large batch inserts or deletes
* Bulk partition operations
* Index rebuild/defragmentation
* Significant data skew changes

### Summary Table

| Statistic         | Meaning                   | Impact on CBO            |
| ----------------- | ------------------------- | ------------------------ |
| BLEVEL            | Branch levels             | Affects depth/I/O cost   |
| LEAF_BLOCKS       | Leaf block count          | Full index scan cost     |
| DISTINCT_KEYS     | Unique key count          | Selectivity decision     |
| CLUSTERING_FACTOR | Row ordering              | Range scan vs. full scan |
| NUM_ROWS          | Table row count           | Cardinality estimation   |
| DENSITY           | Column value distribution | Predicate selectivity    |
| HISTOGRAMS        | Skew of values            | Advanced costing         |
| INDEX_TYPE        | Type of index             | Algorithm availability   |

If you want, I can generate:

* a mermaid diagram showing CBO decision flow
* examples comparing good vs. bad clustering factor
* difference between table statistics vs. index statistics
