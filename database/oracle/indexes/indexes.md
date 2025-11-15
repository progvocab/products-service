Oracle supports a wide range of **index types**, each designed to optimize different kinds of queries and data patterns. Here’s a breakdown of the **most commonly used index types in Oracle**, along with their **use cases**:



### **1. B-tree Index (Default)**
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


### **2. Bitmap Index**
- **Structure**: Uses bitmap vectors for each key value
- **Use Case**: Ideal for columns with **low cardinality** (few distinct values) like gender, status flags
- **Example**:
  ```sql
  CREATE BITMAP INDEX idx_gender ON employee(gender);
  ```
- **Note**: Not suitable for high-concurrency environments (DML-heavy workloads) as it locks the entire range on every Insert / Update.



### **3. Unique Index**
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



### **5. Function-Based Index**
- **Use Case**: Index on expressions or functions to speed up computed WHERE clauses
- **Example**:
  ```sql
  CREATE INDEX idx_upper_name ON employee(UPPER(name));
  ```



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



