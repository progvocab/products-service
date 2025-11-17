Oracle supports a wide range of **index types**, each designed to optimize different kinds of queries and data patterns. Here’s a breakdown of the **most commonly used index types in Oracle**, along with their **use cases**:



## B-tree Index (Default)
- **Structure**: Balanced tree structure
- **Use Case**: Fast lookup on columns with high cardinality (many distinct values)
- **Example**:
  ```sql
  CREATE INDEX idx_employee_name ON employee(name);
  ```
Primary Key is also a B-tree Index.
B-tree indexes are ideal for:
- Fast point lookups
- Range scans
- Sorted access
- Uniqueness enforcement
- High concurrency
- Frequent inserts/updates


A B-tree index in Oracle is a **balanced search tree** with 3 main structures:

### **Root Block**

* Starting point of the index.
* Contains pointers to branch blocks.

### **Branch Blocks**

* Guide the search path.
* Contain key ranges + pointers to deeper blocks.

### **Leaf Blocks**

* Contain the actual **indexed key + ROWID**.
* Sorted in ascending order.
* Linked as a doubly linked list (for fast range scans).

###  Index Lookup Steps

1. Start at **root**.
2. Traverse through **branch blocks** based on key.
3. Reach **leaf block** containing matching key(s).
4. Use **ROWID** to fetch rows from the table.

Always **O(log n)** search complexity.



### B-tree Index Structure**

```mermaid
flowchart TD

    A[Root Block] --> B1[Branch Block 1]
    A --> B2[Branch Block 2]

    B1 --> C1[Leaf Block 1\n(key1..key100)]
    B1 --> C2[Leaf Block 2\n(key101..key200)]

    B2 --> C3[Leaf Block 3\n(key201..key300)]
    B2 --> C4[Leaf Block 4\n(key301..key400)]

    C1 <--> C2
    C2 <--> C3
    C3 <--> C4
```




* B-tree = **root → branch → leaf**.
* Leaf blocks contain **sorted keys + ROWIDs**.



More 
- Why B-tree handles high-concurrency updates safely
- How index block splitting works during inserts


## Bitmap Index**
- **Structure**: Uses bitmap vectors for each key value
- **Use Case**: Ideal for columns with **low cardinality** (few distinct values) like gender, status flags
- **Example**:
  ```sql
  CREATE BITMAP INDEX idx_gender ON employee(gender);
  ```
- **Note**: Not suitable for high-concurrency environments (DML-heavy workloads) as it locks the entire range on every Insert / Update.



##  Unique Index**
- **Use Case**: Ensures uniqueness on one or more columns
- **Example**:
  ```sql
  CREATE UNIQUE INDEX idx_emp_id ON employee(emp_id);
  ```



### **4. Composite Index (Concatenated Index)**
- **Use Case**: Index on multiple columns (order matters)
- **Example**:
  ```sql
  CREATE INDEX idx_emp_name_dept ON employee(last_name, department_id);
  ```



##  Function-Based Index**
- **Use Case**: Index on expressions or functions to speed up computed WHERE clauses
- **Example**:
  ```sql
  CREATE INDEX idx_upper_name ON employee(UPPER(name));
  ```



A functional index is an index created on an expression instead of a physical column. It allows Oracle's Cost-Based Optimizer (CBO) to use an index even when the query applies a function to the column.

### Why Functional Index Is Needed

When a function is applied to a column, Oracle cannot use a normal B-tree index because the stored values in the index differ from the function-applied values.
A functional index stores the computed expression, enabling CBO to choose an INDEX RANGE SCAN instead of a TABLE ACCESS FULL.

### Basic Example

Table:

```sql
CREATE TABLE customers (
  id NUMBER,
  name VARCHAR2(100)
);
```

Query that cannot use a normal index:

```sql
SELECT * FROM customers WHERE UPPER(name) = 'JOHN';
```

Create functional index:

```sql
CREATE INDEX idx_cust_upper_name ON customers (UPPER(name));
```

After creation, CBO performs:

* Expression evaluation using the stored function result
* INDEX RANGE SCAN instead of TABLE ACCESS FULL

### Example With TRIM

Query:

```sql
SELECT * FROM customers WHERE TRIM(phone) = '8888888888';
```

Index:

```sql
CREATE INDEX idx_phone_trim ON customers (TRIM(phone));
```

CBO can now pick IDX_PHONE_TRIM and avoid full table scan.

### Example With CASE Expression

```sql
CREATE INDEX idx_order_status 
ON orders (
  CASE WHEN status IN ('PENDING','CONFIRMED') THEN 1 ELSE 0 END
);
```

Query:

```sql
SELECT COUNT(*)
FROM orders
WHERE CASE WHEN status IN ('PENDING','CONFIRMED') THEN 1 ELSE 0 END = 1;
```

CBO resolves the CASE expression to a stored indexed value and uses INDEX RANGE SCAN.

### Multi-Column Functional Index Example

```sql
CREATE INDEX idx_fullname
ON employees (LOWER(first_name || ' ' || last_name));
```

Query:

```sql
SELECT *
FROM employees
WHERE LOWER(first_name || ' ' || last_name) = 'john doe';
```

CBO uses the stored concatenated lowercase value.

### Requirements and Internal Behavior

Functional indexing requires Oracle to evaluate the expression during DML:

* Oracle Kernel evaluates the functional expression per row during INSERT/UPDATE.
* Index maintenance logic inserts the computed value into the B-tree segment.
* CBO uses the function result during predicate evaluation.

Required session settings (usually default):

```sql
ALTER SESSION SET query_rewrite_enabled = TRUE;
ALTER SESSION SET query_rewrite_integrity = TRUSTED;
```

### How to Verify the Functional Index Is Used

```sql
EXPLAIN PLAN FOR
SELECT * FROM customers WHERE UPPER(name) = 'JOHN';

SELECT * FROM TABLE(DBMS_XPLAN.DISPLAY);
```

Expected operations in the plan:

* INDEX RANGE SCAN on IDX_CUST_UPPER_NAME
* No TABLE ACCESS FULL

### When Not to Use Functional Index

* When the function result has low cardinality (CBO may ignore the index)
* When expressions change frequently and increase DML cost
* When virtual columns referencing the same expression are preferable

If you want, I can generate a comparison between functional index and virtual column index or rewrite for a join optimization scenario.


### **6. Reverse Key Index**
- **Use Case**: Reverses bytes of the key to avoid index hot spots (useful for sequential inserts)
- **Example**:
  ```sql
  CREATE INDEX idx_rev_emp_id ON employee(emp_id) REVERSE;
  ```



### **7. Domain Index**
- **Use Case**: Custom index for complex data types like spatial, text, XML
- **Example**:
  Oracle Text, Oracle Spatial use this under the hood.



### **8. Clustered Index (IOT - Index Organized Table)**
- **Use Case**: Table data stored in index structure itself; no separate table heap
- **Example**:
  ```sql
  CREATE TABLE employee (
    emp_id NUMBER PRIMARY KEY,
    name VARCHAR2(50)
  ) ORGANIZATION INDEX;
  ```

---

### **9. Global vs Local Partitioned Indexes**
- **Local Index**: One index per partition (partitioned table)
- **Global Index**: One index across all partitions



| Index Type         | Best For                         | Notes |
|--------------------|----------------------------------|-------|
| B-tree             | High-cardinality queries         | Default and general-purpose |
| Bitmap             | Low-cardinality, analytical use  | Not good for OLTP |
| Unique             | Enforcing uniqueness             | Same as B-tree with unique constraint |
| Composite          | Multi-column lookups             | Column order matters |
| Function-based     | Queries using expressions        | Must use deterministic functions |
| Reverse Key        | Avoiding hot index blocks        | Useful in high-insert environments |
| Domain             | Custom data types (text/spatial) | Needs special libraries |
| IOT (Clustered)    | Faster primary key access        | Entire row is in the index |
| Partitioned Index  | Partitioned tables               | Local and global variants |



## **Oracle Text**
 (formerly Context Option)

It is Oracle’s built-in full-text search engine, similar to ElasticSearch / Solr features.
Oracle Text can index:

* Plain text columns (`CLOB`, `VARCHAR2`, `NVARCHAR2`)
* PDF, Word, Excel, HTML, XML (using filters)
* JSON
* Binary documents (via filtering)





### `CONTEXT` index

✔ Best for large documents
✔ Supports linguistic search, stemming, fuzzy search
✔ Used for full-text search queries


```sql
CREATE INDEX idx_doc_text ON documents(content)
  INDEXTYPE IS CTXSYS.CONTEXT;
```



### `CTXRULE` index

✔ For classifying documents
✔ Used in message routing/filtering systems



###  `CTXCAT` index

✔ For small documents
✔ Supports structured + text combined queries
(e.g., product search with filters)



Oracle introduces special operators `CONTAINS(column, 'search query') > 0`


```sql
SELECT id, title
FROM documents
WHERE CONTAINS(content, 'oracle AND indexing') > 0;
```



### **Features**

Oracle Text includes:

| Feature                                     | Supported? |
| ------------------------------------------- | ---------- |
| Tokenization                                | ✔          |
| Stemming (e.g., index → indexed → indexing) | ✔          |
| Fuzzy Search                                | ✔          |
| Wildcards                                   | ✔          |
| Linguistic analysis                         | ✔          |
| Stop words                                  | ✔          |
| Synonyms                                    | ✔          |
| Scoring / ranking                           | ✔          |
| Highlight snippets                          | ✔          |
| Document section searching                  | ✔          |
| JSON, XML search                            | ✔          |



### Architecture

Oracle Text uses Inverted index similar to search engine technology. Index stored inside Oracle tablespaces not external system.Automatic background jobs for index maintenance. It is highly optimized and production-grade.Oracle Text integrates deeply with SQL:

* You can combine it with joins
* Use text scoring via `score(1)`
* Search multiple columns using `MULTI_COLUMN_DATASTORE`



```sql
SELECT id, title, score(1) relevance
FROM documents
WHERE CONTAINS(content, 'cloud computing', 1) > 0
ORDER BY relevance DESC;
```

Provided through the **Oracle Text** feature supports rich search features similar to ElasticSearch, uses inverted indexes (`CONTEXT`, `CTXRULE`, `CTXCAT`)



