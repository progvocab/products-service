Exactly ✅ — you’ve spotted the **physical storage structure** of PostgreSQL at the file-system level, and you’re absolutely right.

Let’s break down what you’re seeing in the `base/` directory, what each file represents, and specifically what the `_fsm` and `_vm` files are.

---

## 🧱 1. Directory structure recap

Each folder under:

```
.../base/
```

represents **one database**, named by its **OID** (object identifier).
For example:

```
/opt/homebrew/var/postgres/base/
├── 1/        ← template1
├── 16384/    ← postgres
└── 24576/    ← your user database
```

Each folder (like `24576/`) contains files representing **tables, indexes, TOAST tables**, etc.

---

## 📄 2. Inside a database folder

Example:

```
/opt/homebrew/var/postgres/base/24576/
├── 16384
├── 16384_fsm
├── 16384_vm
├── 16385
├── 16385_fsm
├── 16385_vm
└── ...
```

Here:

| File                | Meaning                                                                                         |
| ------------------- | ----------------------------------------------------------------------------------------------- |
| `16384`             | Main heap (table) file storing actual rows                                                      |
| `16384_fsm`         | **Free Space Map (FSM)** — tracks free space availability in each page                          |
| `16384_vm`          | **Visibility Map (VM)** — tracks which pages have only visible tuples (no unvacuumed dead rows) |
| `16385`             | Another table or index file                                                                     |
| `16385_fsm` / `_vm` | FSM and VM for that table or index                                                              |

---

## 🧩 3. What FSM and VM actually do

### 🧾 **Free Space Map (FSM)**

* Purpose: Helps PostgreSQL quickly find pages that have enough free space for new rows or updates.
* Each table (heap) has one `_fsm` file.
* FSM entries correspond to **pages in the heap**.
* Without it, PostgreSQL would have to scan many pages to find where to insert new rows.

👉 It speeds up **INSERT** and **UPDATE** operations.

---

### 👁️ **Visibility Map (VM)**

* Purpose: Tracks which pages contain only tuples visible to all transactions (no dead tuples).
* Used mainly by **autovacuum** and **index-only scans**.
* A page marked as “all visible” in the VM means:

  * Autovacuum can skip it.
  * Index-only scans can read directly from the index without checking the heap page.

👉 It speeds up **VACUUM** and **index-only SELECT** performance.

---

## ⚙️ 4. Relation between them

| File type | Exists for    | Used by                               | Updated by           |
| --------- | ------------- | ------------------------------------- | -------------------- |
| `_fsm`    | Table & index | Insert path                           | Heap management code |
| `_vm`     | Table only    | Autovacuum, planner, index-only scans | VACUUM, heap pruning |

---

## 📚 5. Optional: Mapping numbers to table names

You can map those numeric file names to real table names with this query:

```sql
SELECT relfilenode, relname, relkind
FROM pg_class
WHERE relkind IN ('r','i')  -- r=table, i=index
ORDER BY relfilenode;
```

Output example:

| relfilenode | relname        | relkind |
| ----------- | -------------- | ------- |
| 16384       | employees      | r       |
| 16385       | employees_pkey | i       |

So file `16384`, `16384_fsm`, `16384_vm` correspond to the **employees** table.

---

## ✅ Summary

| File                       | Description                                           |
| -------------------------- | ----------------------------------------------------- |
| `<oid>`                    | Table or index main data file                         |
| `<oid>_fsm`                | Free Space Map — tracks available space per page      |
| `<oid>_vm`                 | Visibility Map — tracks which pages are fully visible |
| `pg_class` → `relfilenode` | Maps those OIDs to real table/index names             |

---

Would you like me to **draw a labeled diagram** showing how a heap table connects to its FSM and VM files at the page level (with page IDs and what metadata they hold)? It’s a great visual to solidify this.

# Pages 
Perfect follow-up 👏 — and yes, you’re right to think that “pages” are part of a file.

Let’s go deep into what a **page** is in PostgreSQL.

---

## 🧱 1. What is a Page?

A **page** (sometimes called a **block**) is the **smallest unit of I/O and storage** used by PostgreSQL.

* Every table, index, and system catalog is stored in one or more **files**.
* Each of these files is divided into fixed-size **pages**.
* PostgreSQL reads and writes data **one page at a time** — not individual rows or bytes.

---

## 📏 2. Page size

* Default page size: **8 KB (8192 bytes)**
* Can be changed **only at compile time**, not dynamically.
* So every `.data` file (like `16384`, `16384_fsm`, etc.) is internally a sequence of 8KB pages.

You can confirm the block size using SQL:

```sql
SHOW block_size;
```

Typical output:

```
8192
```

---

## 📂 3. File → Pages → Tuples (Rows)

Let’s visualize it:

```
File: 16384  (represents one table)
├── Page 0
│    ├── Header
│    ├── Tuple 1 (row)
│    ├── Tuple 2 (row)
│    └── ...
├── Page 1
│    ├── Header
│    ├── Tuples...
│    └── ...
└── Page N
```

So a **table file** (like `16384`) is made of pages, and **each page holds multiple rows** (tuples).

---

## 🧩 4. Why Pages?

Pages serve several purposes:

| Purpose                 | Description                                                         |
| ----------------------- | ------------------------------------------------------------------- |
| **I/O efficiency**      | PostgreSQL reads/writes 8KB at once to reduce disk I/O overhead     |
| **Buffer management**   | Pages map directly to shared buffers in memory (`shared_buffers`)   |
| **Transaction control** | Each page has headers that track LSN (WAL position), checksum, etc. |
| **Vacuum support**      | Each page knows which rows are visible/dead                         |
| **Concurrency**         | MVCC visibility is tracked per tuple within a page                  |

---

## 🧠 5. Internal Structure of a Page

Each page is divided into several sections:

```
+-----------------------------+
| Page Header                 |
| (LSN, checksum, flags...)   |
+-----------------------------+
| Line Pointer Array          |
| (Offsets for tuples)        |
+-----------------------------+
| Free Space                  |
+-----------------------------+
| Tuple Data (rows)           |
+-----------------------------+
```

When you insert a row, PostgreSQL:

1. Chooses a page with enough free space (via FSM)
2. Adds a line pointer in the array
3. Writes the tuple data at the end
4. Updates page header metadata

---

## ⚙️ 6. Relationship Between File and Pages

| Concept          | Description                                              |
| ---------------- | -------------------------------------------------------- |
| **File**         | One table or index                                       |
| **Page**         | 8KB block within the file                                |
| **Tuple**        | Actual row data inside the page                          |
| **FSM/VM files** | Contain summaries about pages (free space or visibility) |

---

## 🧮 7. Example: Physical storage of a table

Suppose your table has 100 MB of data.

* Each page = 8 KB
* Pages needed = 100 MB / 8 KB = **12,800 pages**
* Those pages will be stored across one or more physical files (since one file can max out at 1 GB by default).

File naming example:

```
16384        → first 1GB segment
16384.1      → next 1GB
16384.2      → next 1GB
...
16384_fsm    → free space map (per-page summary)
16384_vm     → visibility map (per-page summary)
```

---

### ✅ **In summary**

| Concept                            | Description                                                   |
| ---------------------------------- | ------------------------------------------------------------- |
| **Page (block)**                   | Smallest storage & I/O unit in PostgreSQL (default 8 KB)      |
| **Pages live inside files**        | Each table/index = one or more files, each divided into pages |
| **Each page stores tuples (rows)** | With header, line pointers, and free space                    |
| **FSM & VM**                       | Store per-page metadata (free space and visibility)           |
| **I/O operations**                 | Always happen at page (block) level                           |

---

Would you like me to show a **diagram of one heap file** showing its pages, tuple layout, and how FSM and VM summarize those pages? It’s very helpful to visualize the physical storage.

Excellent — you’re now exploring the **complete physical layout of a PostgreSQL database cluster on disk**, which is a crucial part of understanding how PostgreSQL stores and manages data internally.

Let’s go through this in a structured way 👇

---

## 🗂️ 1. Overview — What’s Inside the Data Directory

PostgreSQL’s **data directory** (e.g., `/var/lib/postgresql/data` or `/opt/homebrew/var/postgres`) holds *everything* the cluster needs:

* Databases
* Tables, indexes, metadata
* Configuration files
* Transaction logs
* Caches and status data

It looks like this (simplified):

```
/var/lib/postgresql/data/
├── base/                     ← actual database data (per-database directories)
├── global/                   ← cluster-wide metadata tables
├── pg_xact/                  ← transaction commit status
├── pg_multixact/             ← shared-row lock management
├── pg_wal/                   ← Write-Ahead Log (WAL)
├── pg_stat/                  ← runtime stats
├── pg_subtrans/              ← subtransaction tracking
├── pg_tblspc/                ← tablespace symlinks
├── pg_twophase/              ← 2PC transaction files
├── pg_commit_ts/             ← optional commit timestamps
├── pg_replslot/              ← replication slots
├── pg_logical/               ← logical replication data
├── pg_dynshmem/              ← dynamic shared memory files
├── pg_notify/                ← LISTEN/NOTIFY queues
├── pg_serial/                ← serializable transaction info
├── pg_snapshots/             ← exported snapshots (for long txns)
├── postgresql.conf
├── pg_hba.conf
├── pg_ident.conf
└── pg_control                 ← cluster control file
```

Let’s break these down.

---

## 🧱 2. Important Directories and Their Roles

| Directory         | Purpose                                                 | Key Contents                                                                                                      |
| ----------------- | ------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **base/**         | Holds all per-database directories.                     | Each subdirectory corresponds to one database (`oid` in `pg_database`), containing that DB’s table & index files. |
| **global/**       | Cluster-wide system catalogs (shared across all DBs).   | Files for `pg_authid`, `pg_database`, etc.                                                                        |
| **pg_xact/**      | Transaction commit status log (CLOG).                   | Keeps track of whether transactions are committed, aborted, or in progress.                                       |
| **pg_multixact/** | Multi-transaction data for shared-row locks.            | Used when multiple transactions hold shared locks on same row.                                                    |
| **pg_wal/**       | Write-Ahead Logs.                                       | The redo logs written before data pages are changed. Critical for crash recovery.                                 |
| **pg_subtrans/**  | Subtransaction parent tracking.                         | Used for nested transactions.                                                                                     |
| **pg_tblspc/**    | Symlinks to other filesystem locations for tablespaces. | Each link points to a directory elsewhere.                                                                        |
| **pg_twophase/**  | Files for prepared 2PC transactions.                    | Each file holds state of a prepared transaction.                                                                  |
| **pg_commit_ts/** | Optional commit timestamps (if enabled).                | Used to record exact commit times.                                                                                |
| **pg_dynshmem/**  | Dynamic shared memory segments.                         | Used for shared memory between background workers.                                                                |
| **pg_notify/**    | LISTEN/NOTIFY message queues.                           | Handles asynchronous notifications.                                                                               |
| **pg_logical/**   | Logical replication data.                               | Slot information, temporary changesets, etc.                                                                      |
| **pg_replslot/**  | Replication slot state files.                           | Keep replication slot metadata for logical/physical replication.                                                  |
| **pg_stat/**      | Runtime statistics collector.                           | Holds statistical counters (resets at restart).                                                                   |
| **pg_serial/**    | Serializable transaction tracking.                      | Supports serializable isolation level.                                                                            |
| **pg_snapshots/** | Exported snapshot files.                                | Used by long-running read-only transactions.                                                                      |
| **pg_control**    | Single binary file describing cluster control data.     | Includes LSN position, checkpoint, system ID, etc.                                                                |

---

## 📦 3. Inside `base/<database_oid>/`

That’s where your **actual table and index data lives**.

Example:

```
/var/lib/postgresql/data/base/16384/
├── 2613         ← pg_class (heap file)
├── 2613_fsm     ← Free Space Map for pg_class
├── 2613_vm      ← Visibility Map for pg_class
├── 16385        ← user table (heap)
├── 16385_fsm
├── 16385_vm
├── 16386        ← index on that table
├── 16386_fsm
└── 16386_vm
```

So each table or index in that database corresponds to one or more files here.

---

## 🧩 4. Summary of File Types Inside `base/<db_oid>/`

| File Name                     | Type                        | Meaning                                              |
| ----------------------------- | --------------------------- | ---------------------------------------------------- |
| `<relfilenode>`               | **Heap or index main file** | Actual data (8KB pages)                              |
| `<relfilenode>_fsm`           | **Free Space Map**          | Tracks available free space in heap/index pages      |
| `<relfilenode>_vm`            | **Visibility Map**          | Tracks which pages have only visible (frozen) tuples |
| `<relfilenode>.1`, `.2`, etc. | **Segmented extensions**    | Large tables split after 1GB                         |
| `<relfilenode>_init`          | **Init fork**               | Template for unlogged tables                         |
| `<relfilenode>_toast`         | **TOAST table**             | Stores oversized row data                            |
| `<relfilenode>_toast_index`   | **TOAST index**             | Index on TOAST table                                 |

---

## ⚙️ 5. Supporting System Directories (Brief)

| Directory      | Function                                     |
| -------------- | -------------------------------------------- |
| `pg_xact`      | Tracks commit/abort status of transactions   |
| `pg_subtrans`  | Tracks parent-child subtransaction mapping   |
| `pg_multixact` | Shared-row lock participants                 |
| `pg_wal`       | WAL log files (crash recovery + replication) |
| `pg_twophase`  | 2PC prepared transaction state               |
| `pg_replslot`  | Replication slot persistence                 |
| `pg_stat_tmp`  | Temporary runtime stats                      |
| `pg_logical`   | Logical decoding and replication support     |

---

## 🧠 6. Conceptual Diagram

```
Cluster Directory (PGDATA)
│
├── base/
│    ├── <db_oid1>/
│    │     ├── <relfilenode>        ← Heap/Table file
│    │     ├── <relfilenode>_fsm    ← Free Space Map
│    │     ├── <relfilenode>_vm     ← Visibility Map
│    │     ├── <relfilenode>.1      ← Segmented part
│    │     └── <relfilenode>_toast  ← TOAST table
│    └── <db_oid2>/                 ← Another database
│
├── global/                         ← Shared catalogs
├── pg_xact/                        ← Transaction commit log
├── pg_wal/                         ← Write-ahead logs
├── pg_multixact/                   ← Multi-transaction locks
├── pg_twophase/                    ← 2PC prepared states
└── ...                             ← Other system directories
```

---

## ✅ TL;DR

| Category                       | Component                                               | Purpose                                               |
| ------------------------------ | ------------------------------------------------------- | ----------------------------------------------------- |
| **User data**                  | `heap`, `index`, `_fsm`, `_vm`, `TOAST`                 | Actual table/index data                               |
| **Transaction control**        | `pg_xact`, `pg_subtrans`, `pg_multixact`, `pg_twophase` | Tracks commits, subtransactions, shared locks         |
| **Logging / recovery**         | `pg_wal`                                                | Write-ahead logs for durability                       |
| **Cluster metadata**           | `global/`, `pg_control`                                 | Cluster-wide catalogs and control info                |
| **Runtime / replication**      | `pg_stat`, `pg_notify`, `pg_replslot`, `pg_logical`     | Stats, notify queues, replication data                |
| **Tablespace / snapshot mgmt** | `pg_tblspc`, `pg_snapshots`, `pg_serial`                | Alternate storage paths, snapshots, serializable txns |

---

Would you like me to show a **visual map of how all these directories relate to one another and to the in-memory components** (shared buffers, WAL buffers, background writer, etc.)? That’s the next layer of understanding the full PostgreSQL architecture.
