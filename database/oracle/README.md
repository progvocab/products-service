- [Up](..)

# Oracle 
- [Joins](joins.md)
- [Cursor](cursor)
- [Indexes](indexes)
- [Procedure](procedure)
- [Objects](objects.md)
- [ORA files](ora_files.md)
- [DB Link](db_link.md)


### **Oracle SELECT Query — Internal Execution Steps**

When you run:

```sql
SELECT * FROM employees WHERE dept_id = 10;
```

Oracle performs the following steps **in order**:



### **1. Parse Phase (SQL Parsing + Syntax + Semantics + Security)**

### Components involved:

* **Parser**
* **Semantic Checker**
* **Data Dictionary Cache**
* **Library Cache (Shared Pool)**
* **User privileges subsystem**

### What Oracle does:

1. Check SQL syntax.
2. Validate table/column names against data dictionary.
3. Check user privileges.
4. Generate a **parse tree**.

If a similar statement already exists in the shared pool (same text + same binds), Oracle may reuse it (soft parse).



### **2. Optimization Phase (Cost-Based Optimization)**

### Components:

* **Cost-Based Optimizer (CBO)**
* **Statistics subsystem**
* **Metadata: histograms, indexes, partition metadata**

### What happens:

Oracle tries many possible execution paths and chooses the lowest cost plan:

✔ Which index to use
✔ Whether to do full table scan or index scan
✔ Join order
✔ Join method: hash/merge/nested loop
✔ Partition pruning
✔ Predicate pushdown
✔ Whether to use parallel execution

Result → **Execution Plan**

Here is a **very clear, concise, but complete** explanation of **how Oracle’s Cost-Based Optimizer (CBO) finds the “best” execution plan**.


## **How Oracle CBO Chooses the Best Execution Plan**

Oracle’s CBO tries multiple possible ways to execute a query and **assigns a cost to each path**.
The plan with the **lowest cost** is selected.

It uses a 4-stage process:



### **1. Generate All Possible Plans (Search Space Exploration)**

Oracle enumerates all possible **logical and physical transformations**, such as:

### **Join orders**

* T1 → T2 → T3
* T2 → T3 → T1
* Bushy trees, left-deep trees

### **Join methods**

* Nested Loop Join
* Hash Join
* Merge Join

### **Access paths**

* Full table scan
* Index range scan
* Index skip scan
* Fast full index scan
* Partition pruning

### **Predicate pushdown & rewrite rules**

* Filter pushdown
* Subquery unnesting
* Predicate transitivity
* View merging
* Star transformation

These combinations form the **plan search space**.


### **2. Collect Statistics & Metadata (CBO Input Data)**

The optimizer uses metadata from:

### **Object Stats**

* Table row count
* Number of blocks
* Index leaf blocks
* Distinct values per column
* Column selectivity

### **Histograms**

* Frequency histograms (skewed data)
* Height-balanced histograms

### **System Stats**

* CPU speed
* I/O performance
* Multiblock read count

### **Partition Metadata**

* Prunable partitions
* Local/global index ranges

### **Optimizer Parameters**

* `optimizer_mode`
* `optimizer_index_cost_adj`
* `_query_execution_plan` flags

All this allows CBO to *estimate* work required.



### **3. Estimate Cost for Each Operator**

CBO assigns a cost to each step based on:

### **I/O Cost**

* How many table blocks must be read
* How many index leaf blocks
* Sequential read vs random read
* Partition scans vs full scans

### **CPU Cost**

* Predicate evaluation
* Sorting rows
* Hash table creation for hash joins
* Comparing rows in merge joins

### **Cardinality Estimates**

Estimate rows flowing between operators:

```
estimatedRows = totalRows * selectivity
```

Cardinality drives:

* join method selection
* memory allocation (PGA)
* ordering of filters



### **4. Choose Plan With the Lowest Total Cost**

Oracle uses a dynamic programming algorithm:

* Combine operator costs bottom-up
* Compare total cost of each possible plan
* Pick the cheapest plan

This cost is a unitless number representing **estimated work**, not time.

Example:

| Plan                        | Cost |
| --------------------------- | ---- |
| Index scan + nested loop    | 93   |
| Hash join + full table scan | 74   |
| Merge join + index scan     | 112  |

Oracle picks **hash join + full table scan** because **74 is lowest**.



### **What Makes CBO Extremely Powerful**

### ✔ Adaptive Optimization

During execution, Oracle can change the plan if cardinality estimates were wrong (adaptive plan).

### ✔ Bind Variable Peeking

Examines first bind value to adjust plan.

### ✔ Dynamic sampling

Collects sample statistics when needed.

### ✔ SQL Plan Baselines

Remembers stable plans to prevent regressions.



### **Oracle CBO chooses the best plan by:**

1. **Generating many candidate plans**
2. **Using statistics to estimate row counts and cost**
3. **Computing I/O + CPU cost per plan**
4. **Choosing the plan with the lowest estimated cost**

The chosen plan becomes the **execution plan** stored with the cursor.



> More :
📌 Why CBO sometimes picks a bad plan
📌 How to override CBO using hints
📌 How cardinality misestimation happens
📌 How adaptive plans fix the wrong choice

---


### **3. Row Source Generation (Plan → Operators)**

### Components:

* **Row Source Generator**
* **Cursor**

The optimizer’s plan is converted into a **row source tree**, where each node is an operator:

* TABLE ACCESS FULL
* INDEX RANGE SCAN
* HASH JOIN
* SORT
* GROUP BY
* FILTER
* VIEW MERGING nodes

This row source tree becomes the **cursor**, which Oracle will execute.



### **4. Execution Phase (Actual Data Retrieval)**

### Components:

* **Buffer Cache (DB Cache)**
* **Redo Log Buffer**
* **PGA (Private memory)**
* **Segments (Tables, Indexes)**
* **Direct Path mechanisms (if used)**

Steps:

1. Check if requested blocks are in **Buffer Cache** → if yes, return them.
2. If not, read blocks from datafiles (I/O) into buffer cache.
3. Apply filters, conditions, joins.
4. Return rows gradually using pipelining.

Important:

* Oracle uses **cursor iteration** to return rows in batches (fetch calls).
* Each row source operator pulls rows from its children (“demand pull model”).



### **5. Fetch Phase (Return Rows to Client)**

### Components:

* **Cursor**
* **SQL*Net / JDBC / OCI layer**

Oracle sends rows to the client in **multiple fetch calls**, not all at once.

Fetch size determines how many rows per call.



```
Parsing
  ↓
Semantic checks (data dictionary)
  ↓
Shared pool lookup (soft parse?)
  ↓
Optimizer (CBO decides best plan)
  ↓
Row source tree generated (cursor)
  ↓
Execution starts (read blocks, apply filters)
  ↓
Rows fetched in batches via cursor
```



### **Participants**

| Oracle Component                      | Role                                            |
| ------------------------------------- | ----------------------------------------------- |
| **Shared Pool**                       | Stores parsed SQL, execution plans, metadata    |
| **Library Cache**                     | Stores execution plans (cursors)                |
| **Dictionary Cache**                  | Stores metadata (tables, columns, privileges)   |
| **Optimizer (CBO)**                   | Generates the best execution plan               |
| **Row Source Generator**              | Converts plan → executable operators            |
| **Buffer Cache**                      | Holds data blocks read from disk                |
| **PGA**                               | Private memory used for sorting, hashing, joins |
| **Redo Log Buffer**                   | If DML, records changes (not used in SELECT)    |
| **Background Processes (DBWR, LGWR)** | Handle block writes, logging                    |
| **SQL*Net**                           | Sends data back to client                       |





✔ **Parse** → Validate SQL, check metadata, create parse tree
✔ **Optimize** → CBO chooses best execution plan
✔ **Row Source** → Plan converted to operators (cursor)
✔ **Execute** → Blocks read, filters applied
✔ **Fetch** → Results returned to client in batches

This is the full lifecycle of a SELECT query inside Oracle.



If you want, I can also explain:
🔸 How Oracle chooses between index scan vs full table scan
🔸 How joins are executed internally (hash/merge/nested loop)
🔸 How Oracle pipelines row sources during execution
🔸 What happens during a *hard parse* vs *soft parse*



