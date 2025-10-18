That’s a fantastic and deep question — exactly the kind of “under-the-hood” understanding that separates system users from system designers.

Let’s walk step-by-step through what happens inside **PostgreSQL** when a user fires a simple query such as:

```sql
SELECT * FROM employee WHERE id = 1;
```

We’ll go from **SQL text → parser → executor → storage → visibility check**, showing every major internal subsystem that participates.

---

## 🧭 1️⃣ Entry Point: The Client & Postmaster

When you send a query from `psql` or a driver:

1. The **PostgreSQL client library (`libpq`)** sends the SQL command to the server over a TCP connection (port 5432).
2. The **Postmaster process** (the main PostgreSQL daemon) accepts the connection.
3. It **forks a backend process** (`postgres`) dedicated to that client session.

From now on, every query from your session is handled by that backend.

---

## 🧩 2️⃣ The SQL Processing Pipeline

### a. **Parser**

* Component: `parser/analyze.c`, `gram.y`
* Function: Converts the SQL text into a **parse tree**.
* Example:

  ```sql
  SELECT * FROM employee WHERE id = 1;
  ```

  becomes an internal tree:

  ```
  SelectStmt
   ├── targetList → ['*']
   └── whereClause → (id = 1)
  ```

### b. **Analyzer / Rewriter**

* Checks:

  * Column existence
  * Data types
  * Expands `*` into actual columns
* Applies **query rewrite rules** (e.g., views, rules, INSTEAD OF triggers)

### c. **Planner / Optimizer**

* Component: `optimizer/`
* Chooses the best execution plan using:

  * **pg_class** (table stats)
  * **pg_statistic** (column histograms)
  * **pg_index**, **pg_attribute**
* Produces a **Plan Tree**:

  ```
  Index Scan using employee_pkey on employee  (cost=0.29..8.30 rows=1)
   Filter: (id = 1)
  ```

---

## 🧮 3️⃣ Executor (The Heart of Runtime)

The **executor** walks the plan nodes and fetches tuples.

Each node (e.g., Index Scan, Seq Scan, Filter, Sort) is a small state machine implementing:

```
ExecInitNode()   → initialize
ExecProcNode()   → get next tuple
ExecEndNode()    → cleanup
```

So `SELECT * FROM employee WHERE id=1` may look like:

```
[Executor]
    ↓
[IndexScan Node]
    ↓
[Heap Fetch (row version)]
    ↓
[Visibility check (MVCC)]
    ↓
[Return tuple to client]
```

---

## 💾 4️⃣ Access Method Layer (AM Layer)

Two key access methods interact here:

| Layer        | Responsible For                              | Example Code                 |
| ------------ | -------------------------------------------- | ---------------------------- |
| **Index AM** | Search in B-tree, hash, or GiST index        | `src/backend/access/nbtree/` |
| **Heap AM**  | Retrieve physical tuple (row) from heap file | `src/backend/access/heap/`   |

Example flow for an index lookup:

1. The **index** is consulted (`id = 1` → page & tuple offset).
2. It returns a **TID (Tuple Identifier)** — basically `(block number, offset)`.

---

## 📂 5️⃣ Buffer Manager & Shared Memory

The **Buffer Manager** ensures the required table or index page is in shared memory.

* Uses **Buffer Pool**: fixed-size cache of data pages.
* If page not in memory → reads it from disk using the **Storage Manager (SMGR)**.

```
Shared Buffers
 ├── Page 1 [employee heap]
 ├── Page 2 [employee index]
 └── ...
```

---

## 💿 6️⃣ Storage Manager (SMGR) & File System

* Each table is stored as one or more files in `base/<db_oid>/<relfilenode>`.
* SMGR performs actual **read/write syscalls** on these files.
* On a read:

  * SMGR asks the **OS page cache** for the block.
  * If not cached, the kernel reads it from disk (HDD or SSD).

---

## 🔐 7️⃣ Transaction Manager & MVCC

Once the tuple (row) is in memory, PostgreSQL performs **MVCC visibility check**:

| Field  | Meaning                             |
| ------ | ----------------------------------- |
| `xmin` | XID that inserted this tuple        |
| `xmax` | XID that deleted/updated this tuple |

It compares these against the transaction’s **snapshot** to decide if the row is visible (as explained earlier).

---

## 🧹 8️⃣ Vacuum, Free Space & Visibility Maps

* **Visibility Map (VM):** tells if all rows on a page are visible (for index-only scans).
* **Free Space Map (FSM):** tracks pages with room for new rows.
* **Vacuum** cleans old, invisible tuples later.

---

## 📨 9️⃣ Tuple to Output

Once the tuple passes visibility checks:

1. It is returned up the executor nodes.
2. The executor formats the tuple into a result row.
3. The backend encodes it (text/binary) via the **PostgreSQL FE/BE protocol**.
4. Sent back over the TCP socket to the client.

---

## 🧠 10️⃣ Summary — Internal Components Involved

| Component                        | Responsibility                           |
| -------------------------------- | ---------------------------------------- |
| **Postmaster**                   | Accepts client connection, forks backend |
| **Parser / Analyzer / Rewriter** | SQL parsing & rule application           |
| **Planner / Optimizer**          | Generates best execution plan            |
| **Executor**                     | Executes plan nodes, returns tuples      |
| **Index Access Method**          | Locates tuples by key                    |
| **Heap Access Method**           | Reads and manages physical tuples        |
| **Buffer Manager**               | Caches disk pages in shared memory       |
| **Storage Manager (SMGR)**       | Performs physical I/O                    |
| **Transaction Manager**          | Assigns XIDs, tracks commits             |
| **MVCC System**                  | Decides visibility of tuples             |
| **WAL / Checkpointer**           | Ensures durability and crash recovery    |
| **VACUUM / FSM / VM**            | Cleans up and maintains heap space       |

---

## 🧩 Simplified Text Diagram

```
Client
  │
  ▼
Postmaster ──► Backend Process
  │
  ▼
[Parser] → [Analyzer] → [Planner] → [Executor]
                                    │
                                    ▼
                             [Access Methods]
                             /              \
                    [Index AM]             [Heap AM]
                           │                    │
                           ▼                    ▼
                  [Buffer Manager]       [Transaction Manager]
                           │                    │
                           ▼                    ▼
                        [SMGR]               [MVCC Check]
                           │
                           ▼
                         [Disk]
```

---

## ⚡ Quick Analogy

Think of PostgreSQL like a layered operating system:

| OS Concept        | PostgreSQL Equivalent        |
| ----------------- | ---------------------------- |
| Virtual Memory    | Buffer Manager               |
| Scheduler         | Query Planner                |
| File System       | Storage Manager              |
| Process Table     | Transaction Manager          |
| Garbage Collector | VACUUM                       |
| Syscalls          | Access Methods (Heap, Index) |

---

Would you like me to show a **Mermaid sequence diagram** of this entire “SELECT path” — from client query → buffer manager → disk → back to client — to visualize how these components interact?


Perfect — this is one of the best ways to understand **PostgreSQL’s internal architecture** and how data is organized and read.
Let’s visualize the **hierarchical structure** of storage and how components like **WAL**, **Heap**, and **Indexes** relate.

---

## 🧠 Overview: PostgreSQL Internal Storage & Read Path

Below is a **hierarchical diagram** (text-based) showing relationships between all major storage structures:

```
                        +----------------------------------+
                        |        PostgreSQL Instance       |
                        +----------------------------------+
                                      |
         +----------------------------+-----------------------------+
         |                                                          |
+-----------------------+                               +-----------------------+
|      Shared Buffers   |                               |   Write-Ahead Log     |
| (In-memory Cache Area)|                               |        (WAL)          |
+-----------------------+                               +-----------------------+
         |                                                          |
         |                                                          |
         V                                                          V
+--------------------------------------------------------------------------+
|                              Data Directory                              |
|                (e.g., /var/lib/postgresql/data/base/)                    |
+--------------------------------------------------------------------------+
   |                          
   +--> Database (e.g., mydb/)
          |
          +--> Table (e.g., employees)
                 |
                 +--> Table File (Heap) — stores actual rows
                 |        |
                 |        +--> Pages (8 KB blocks)
                 |               |
                 |               +--> Tuples (rows / row versions)
                 |                        |
                 |                        +--> Each tuple has xmin/xmax (MVCC)
                 |
                 +--> Free Space Map (FSM) — tracks free space in pages
                 |
                 +--> Visibility Map (VM) — tracks all-visible/all-frozen pages
                 |
                 +--> Index File(s)
                          |
                          +--> B-Tree / GIN / GiST etc.
                          |        |
                          |        +--> Index Pages (8 KB)
                          |                 |
                          |                 +--> Index Entries (key + TID pointer)
                          |
                          +--> TID (Tuple ID) → Points to specific row in Heap
```

---

## ⚙️ Step-by-Step: What Happens on `SELECT * FROM employees WHERE id = 10`

Let’s go through the **read path**:

1. **SQL Parser & Planner**

   * The query goes through the **parser**, then the **planner/optimizer** determines:

     * Which **index** (if any) to use.
     * Which **access path** (sequential scan, index scan, bitmap scan) is best.

2. **Executor Begins Scan**

   * The **executor** requests pages from the table or index.
   * If an **index** is used:

     * The **index file** (e.g., B-Tree) is read first.
     * Index entry points to the heap tuple using a **TID (block#, offset)**.

3. **Buffer Manager / Shared Buffers**

   * PostgreSQL checks if the required page (heap or index) is already in **shared buffers**.

     * ✅ If yes → use it (cache hit)
     * ❌ If no → load from disk using **OS read()**

4. **Heap Page Read**

   * The page (8 KB block) from the heap is fetched.
   * Within the page, PostgreSQL finds the tuple using **TID** (e.g., block 12, offset 5).

5. **MVCC Visibility Check**

   * PostgreSQL checks tuple’s **xmin/xmax** fields:

     * Is this version visible to the current transaction snapshot?
     * If not visible (e.g., deleted or updated), check previous/next versions.

6. **Return Visible Tuple**

   * The visible tuple’s data is read from the heap and passed back to the executor.

7. **Executor Produces Result**

   * The executor returns the resulting rows to the client.

---

## 🧩 Relation to WAL (Write-Ahead Log)

Even during reads, the **WAL** plays an indirect role:

* It ensures all prior writes that affect the visible data are **durably stored**.
* When PostgreSQL restarts, it replays WAL to restore the heap and indexes to a consistent state.

---

## 📦 Summary Table

| Component                | Description                               | Size                      | Role                           |
| ------------------------ | ----------------------------------------- | ------------------------- | ------------------------------ |
| **Heap**                 | Main table file storing tuples            | 8 KB pages                | Stores actual data             |
| **Index**                | Secondary structure mapping key → TID     | 8 KB pages                | Accelerates lookup             |
| **FSM (Free Space Map)** | Tracks free space in heap pages           | —                         | Helps find space for inserts   |
| **VM (Visibility Map)**  | Tracks pages where all tuples are visible | —                         | Helps VACUUM skip pages        |
| **WAL**                  | Write-ahead log of all changes            | —                         | Ensures durability             |
| **Shared Buffers**       | Memory cache of pages                     | Configurable (e.g., 1 GB) | Reduces disk I/O               |
| **Tuple xmin/xmax**      | Transaction IDs                           | —                         | Enable MVCC visibility control |

---

Would you like me to add a **diagram showing how MVCC row versions are chained inside a heap page** (i.e., how updates create new tuples and maintain visibility)?
It would perfectly complement this structure.

