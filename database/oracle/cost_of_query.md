

## Query Cost 

**Query cost** is a number that represents the estimated **resource usage** Oracle expects for executing a SQL query, calculated by the **Cost-Based Optimizer (CBO)**.

> **Cost is unitless**
> It’s an abstract metric influenced by I/O, CPU, memory usage, and more—not actual execution time.



| Concept | Description |
|--------|-------------|
| **CBO (Cost-Based Optimizer)** | Determines the most efficient execution plan using statistics. |
| **Execution Plan** | The step-by-step method Oracle uses to execute a SQL query. |
| **Statistics** | Metadata about tables, indexes, data distribution—crucial for CBO decisions. |
| **Cardinality** | The estimated number of rows returned by each operation. |
| **Selectivity** | Fraction of rows a predicate returns (lower = more selective = better). |



Find the Query Cost in Oracle using `EXPLAIN PLAN`

```sql
EXPLAIN PLAN FOR
SELECT * FROM employees WHERE department_id = 10;

SELECT * FROM TABLE(DBMS_XPLAN.DISPLAY);
```

 `AUTOTRACE` in SQL*Plus

```sql
SET AUTOTRACE ON
SELECT * FROM employees WHERE department_id = 10;
```

Shows:
- Execution plan
- Statistics (logical reads, physical reads, etc.)
- Cost

### Oracle SQL Developer

1. Paste the query.
2. Press **F10** or click **Explain Plan**.
3. View the **cost**, access paths (full table scan, index), and rows.



## Reading the Execution Plan

| Column        | Meaning |
|---------------|---------|
| `Operation`   | Action being performed (e.g., TABLE ACCESS FULL, INDEX RANGE SCAN) |
| `Options`     | Modifiers for operation (e.g., FULL, BY ROWID) |
| `Object Name` | Table or index used |
| `Cost`        | Estimated cost of the operation |
| `Cardinality` | Estimated rows returned |
| `Bytes`       | Estimated size of rows returned |



##  Performance Tuning Techniques

### Use Proper Indexes

```sql
CREATE INDEX idx_dept_id ON employees(department_id);
```
This allows Oracle to use `INDEX RANGE SCAN` instead of full table scans.

### Where clause

adding a WHERE clause usually improves performance because it allows the database to filter rows earlier, reducing the amount of data that needs to be scanned, joined, sorted, or returned. When a WHERE condition matches an indexed column, the optimizer can use the index, which avoids a full table scan and directly accesses only the needed rows.

### Union to union all


UNION removes duplicates → requires extra work** `UNION` performs:

1. Combine result sets
2. Sort or hash results
3. Remove duplicate rows

This requires:

* Sorting large datasets
* Extra CPU
* Extra memory (TEMP tablespace in Oracle)
* Sometimes disk spill

###  UNION ALL does not remove duplicates**

`UNION ALL` simply **appends the results** of both queries.
No sorting.
No hashing.
No duplicate elimination.
No TEMP tablespace usage.

**Performance Impact**

* **Much faster** (sometimes **10x–50x** depending on row counts)
* **Less CPU and Memory**
* **No temporary segment allocation**
* **No extra sorting overhead**



when NOT to use UNION ALL
Only when you **must eliminate duplicates** between the result sets.



###  Update Statistics

```sql
EXEC DBMS_STATS.GATHER_TABLE_STATS('HR', 'EMPLOYEES');
```
Ensures optimizer has accurate information.

###  Use Bind Variables

```sql
SELECT * FROM employees WHERE department_id = :dept;
```
Improves **cursor reuse** and avoids hard parsing.

###  Avoid SELECT * (fetch only required columns)

```sql
SELECT name, salary FROM employees WHERE department_id = 10;
```

### Use EXISTS instead of IN

```sql
-- Better:
SELECT name FROM employees e WHERE EXISTS (
  SELECT 1 FROM departments d WHERE e.dept_id = d.id AND d.region = 'EMEA'
);
```

### Use Partitioning (for very large tables)

```sql
CREATE TABLE sales (
  id NUMBER,
  sale_date DATE,
  amount NUMBER
)
PARTITION BY RANGE (sale_date) (
  PARTITION p2024 VALUES LESS THAN (TO_DATE('01-JAN-2025', 'DD-MON-YYYY')),
  PARTITION p2025 VALUES LESS THAN (TO_DATE('01-JAN-2026', 'DD-MON-YYYY'))
);
```


## Advanced Techniques 

| Technique | Benefit |
|----------|---------|
| **Materialized Views** | Precompute and cache complex joins/aggregates. |
| **SQL Profiles / SQL Plan Baselines** | Enforce optimal plans. |
| **Parallel Query** | Distribute execution across CPUs. |
| **Histogram Statistics** | Improve accuracy for skewed column values. |



##  Query Cost Analysis

```sql
EXPLAIN PLAN FOR
SELECT first_name, salary FROM employees WHERE department_id = 50;

SELECT * FROM TABLE(DBMS_XPLAN.DISPLAY);
```

**Plan Output:**
```
| Id | Operation          | Name              | Rows | Cost |
|----|--------------------|-------------------|------|------|
| 0  | SELECT STATEMENT   |                   |      | 4    |
| 1  | TABLE ACCESS BY INDEX ROWID | EMPLOYEES | 10   | 4    |
| 2  | INDEX RANGE SCAN   | IDX_DEPT_ID       | 10   | 2    |
```

- **Cost = 4**: Relatively cheap
- Uses **Index Range Scan** = optimized path

---

##  Real-Life Use Case

**Scenario**: HR team runs salary reports, but queries are slow.

**Steps to Tune:**
1. Run `EXPLAIN PLAN`
2. Identify full table scan
3. Create index on `department_id`
4. Gather table stats
5. Rewrite query to avoid `SELECT *`

**Result**: Query cost reduced, execution 10x faster.





| Topic | Summary |
|-------|---------|
| **Query Cost** | Internal estimate by Oracle of resource usage for a query. |
| **Tools** | `EXPLAIN PLAN`, `DBMS_XPLAN`, SQL Developer, `AUTOTRACE`. |
| **Tuning Techniques** | Indexing, statistics, rewrite queries, partitioning. |
| **Goal** | Reduce cost, improve performance, preserve correctness. |





Below is a **clear list of the most common Oracle execution plan operations**, each with a **4–5 line concise explanation**, exactly as you asked.



## **Oracle Execution Plan Operations**


### **TABLE ACCESS FULL**

Oracle reads **every block** of the table sequentially.
Used when no useful index exists, the predicate is not selective, or full scan is cheaper.
Can benefit from **multiblock reads** and **smart scans (Exadata)**.
Often the slowest for OLTP but good for scanning large portions of data.



## **TABLE ACCESS BY INDEX ROWID**

Oracle first finds matching rowids using an **index scan**, then fetches rows from the table.
This means two operations: index access → table access.
Faster than full scans when the predicate is selective.
Used commonly with B-tree indexes.



### **INDEX RANGE SCAN**

Scans part of an ordered B-tree index (start → end).
Occurs when Oracle can apply a predicate on the **leading column(s)** of the index.
Efficient for ranges (`>`, `<`, BETWEEN) and equality matches.
Returns rowids to fetch rows.



### **INDEX UNIQUE SCAN**

Accesses **exactly one** index entry because the index is **unique**.
Very fast, constant-time lookup.
Used for primary key or unique key lookups.
Returns one rowid for table fetch.



### **INDEX FULL SCAN**

Reads the entire index in sorted order.
Used when the index itself contains all needed data (**index-only** query).
Faster than full table scan if the index is smaller than the table.
Can also be chosen for ORDER BY optimization.



### **INDEX FAST FULL SCAN**

Reads the entire index similar to TABLE FULL SCAN but using **multi-block reads**.
Does **not** maintain index order.
Used when the query can be satisfied from index columns alone.
Does not require predicates on index leading column.



### **BITMAP INDEX SCAN**

Used on **low cardinality columns** like STATUS, GENDER, REGION.
Oracle retrieves bitmap vectors instead of rowids.
Very fast for combining multiple bitmap indexes with AND/OR.
Not ideal for OLTP; best for data warehouses.



### **HASH JOIN**

Builds a hash table on the smaller dataset and probes it with the larger one.
Efficient for large, unsorted datasets.
Requires memory (PGA) and may spill to TEMP if insufficient.
Preferred when join columns are not indexed.



### **NESTED LOOPS JOIN**

For each row from the outer table, Oracle probes the inner table (ideally via index).
Very fast for **small outer table + indexed inner table**.
Good for OLTP queries accessing few rows.
Can be slow if both sides are large.



### **MERGE JOIN**

Sorts both tables by join key and merges them.
Ideal when both inputs are already sorted or indexed.
Fast for large datasets when sorting cost is low.
Used for equality joins.


## **FILTER**

Applies a condition to rows flowing through the plan.
Often used for correlated subqueries.
Rows not satisfying the filter predicate are discarded.
Does not inherently imply an index scan.



## **SORT (ORDER BY / GROUP BY / AGGREGATE)**

Oracle sorts rows in memory or TEMP.
Used for `ORDER BY`, `GROUP BY`, `DISTINCT`, analytic functions.
May cause TEMP spills if memory is insufficient.
Sorting is relatively expensive.



## **VIEW**

Represents a subquery or inline view.
Can be merged or materialized depending on optimizer choices.
Acts as a logical container within the plan.
Often used with WITH subqueries.



## **HASH GROUP BY**

Uses hashing instead of sorting to group rows.
Efficient for large, unsorted datasets.
May write to TEMP if hash table doesn’t fit in memory.
Often faster than SORT GROUP BY for large inputs.



## **CONNECT BY / RECURSIVE WITH**

Used for hierarchical queries (tree structures).
Processes parent-child relationships recursively.
Can do depth-first or breadth-first traversal.
Commonly used with HR organizational trees.




More :
or
✅ **Examples showing actual EXPLAIN PLAN outputs**
or
✅ **Detailed performance tuning guide for each operation**




