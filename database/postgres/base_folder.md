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

