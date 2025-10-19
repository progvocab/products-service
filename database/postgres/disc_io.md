Excellent question — this goes right to the **core of PostgreSQL’s storage engine** 🔥

Let’s break it down precisely:
PostgreSQL doesn’t use an *external* C/C++ library for reading or writing its database files — it **implements its own custom storage and I/O layer** directly in C, built on top of standard POSIX system calls (`read()`, `write()`, `open()`, `lseek()`, `fsync()`, etc).

But to understand how PostgreSQL performs I/O efficiently, we’ll look at:

1. The **C libraries** and **system APIs** it depends on.
2. The **internal PostgreSQL modules** that manage file I/O and buffering.
3. The **data flow** from SQL → file system.

---

## 🧩 1. C / POSIX APIs PostgreSQL Uses for I/O

PostgreSQL uses **low-level Unix/POSIX I/O** — no C++ STL or external I/O libraries.

| Function               | Header         | Purpose                                          |
| ---------------------- | -------------- | ------------------------------------------------ |
| `open()`               | `<fcntl.h>`    | Open file descriptors for relation files         |
| `read()`               | `<unistd.h>`   | Read raw bytes from file                         |
| `write()`              | `<unistd.h>`   | Write raw bytes to file                          |
| `lseek()`              | `<unistd.h>`   | Move file pointer to an offset (to locate pages) |
| `fsync()`              | `<unistd.h>`   | Force dirty pages to disk for durability         |
| `fdatasync()`          | `<unistd.h>`   | Flush file data (without metadata)               |
| `pread()` / `pwrite()` | `<unistd.h>`   | Position-based read/write, thread-safe           |
| `mmap()`               | `<sys/mman.h>` | Rarely used; PostgreSQL prefers explicit I/O     |
| `ftruncate()`          | `<unistd.h>`   | Resize relation file                             |
| `stat()`               | `<sys/stat.h>` | Get file information                             |

✅ So PostgreSQL **directly uses these system calls**, wrapped in its own file abstraction layer.

---

## ⚙️ 2. PostgreSQL’s Internal I/O Layer

PostgreSQL defines **its own internal libraries (modules)** in C for reading/writing data files safely and efficiently.

### 🔹 a. `smgr/` — Storage Manager Layer

Located in `src/backend/storage/smgr/`

* Provides an abstract API for reading/writing relation (table/index) blocks.
* Main interface functions:

  * `smgrread()`
  * `smgrwrite()`
  * `smgrextend()`
  * `smgrsync()`
  * `smgrtruncate()`

These call **lower-level file manager (md.c)** functions that use actual system calls.

🗂️ File: `src/backend/storage/smgr/md.c`

Example code:

```c
void mdread(SMgrRelation reln, ForkNumber forknum, BlockNumber blocknum, void *buffer)
{
    off_t seekpos = (off_t) blocknum * BLCKSZ;
    int fd = OpenTemporaryFile(reln, forknum);
    if (lseek(fd, seekpos, SEEK_SET) < 0)
        elog(ERROR, "could not seek to block %u", blocknum);
    if (read(fd, buffer, BLCKSZ) != BLCKSZ)
        elog(ERROR, "could not read block %u", blocknum);
}
```

So PostgreSQL *manually* seeks and reads 8 KB blocks (default `BLCKSZ`).

---

### 🔹 b. `bufmgr/` — Buffer Manager Layer

Located in `src/backend/storage/buffer/`

* Acts as a **cache between the disk and executor**.
* Decides whether to:

  * Read a block from disk via `smgrread()`, or
  * Serve it from shared memory if already cached.

Core functions:

* `ReadBuffer()`
* `ReleaseBuffer()`
* `FlushBuffer()`

It uses the **shared buffer pool** (in shared memory), managed via **LRU-like algorithms**.

---

### 🔹 c. `relfilenode` and Relation Files

Each PostgreSQL table/index maps to one or more physical files on disk:

```
$PGDATA/base/<db_oid>/<relfilenode>
```

These files contain:

* Heap pages (data rows)
* FSM (Free Space Map)
* VM (Visibility Map)

PostgreSQL never memory-maps them; instead, it reads/writes blocks manually via buffer manager.

---

### 🔹 d. `wal/` — Write-Ahead Logging

Located in `src/backend/access/transam/xlog.c`

This subsystem:

* Writes **WAL records** (redo logs) to files using the same low-level `write()` calls.
* Ensures durability via `fsync()` after commit.

---

### 🔹 e. `fd.c` — File Descriptor Cache

PostgreSQL maintains its own **file descriptor cache** (in `src/backend/storage/file/fd.c`) to limit open files and reuse them.

---

## 🔄 3. Data Flow (SQL to Disk)

Let’s see what happens when you run an `INSERT`:

```
SQL INSERT
   ↓
Executor
   ↓
Buffer Manager (bufmgr.c)
   ↓
  [Cache Miss] → calls Storage Manager
                   ↓
                 md.c  (mdwrite, mdread)
                   ↓
           POSIX syscalls (open, write, fsync)
                   ↓
             Filesystem / Kernel Page Cache
                   ↓
                    Disk
```

---

## 🧠 4. C/C++ Library Summary

| Layer                    | Implementation             | Library / API Used                              |
| ------------------------ | -------------------------- | ----------------------------------------------- |
| File access              | PostgreSQL `md.c`          | POSIX: `open()`, `read()`, `write()`, `lseek()` |
| Buffer management        | PostgreSQL `bufmgr.c`      | Shared memory, custom LRU                       |
| WAL logging              | PostgreSQL `xlog.c`        | POSIX + checksums                               |
| File descriptor cache    | PostgreSQL `fd.c`          | Internal C structs                              |
| Compression (TOAST, WAL) | zlib, LZ4, zstd (optional) | External compression libraries                  |
| Encryption (if enabled)  | OpenSSL                    | External library for crypto only                |

---

## ✅ In Summary

PostgreSQL **does not use any external C++ or database library** for reading/writing its data.
Instead, it implements its own **modular storage engine** in C on top of **POSIX system calls**, with:

| Purpose             | Module      | Core Mechanism      |
| ------------------- | ----------- | ------------------- |
| Physical file I/O   | `smgr/md.c` | POSIX syscalls      |
| Buffering & caching | `bufmgr.c`  | Shared memory + LRU |
| Logging             | `xlog.c`    | WAL write & fsync   |
| File reuse          | `fd.c`      | Descriptor cache    |

---

Here’s a **text-based architecture diagram** showing the complete flow of PostgreSQL **I/O path** — from the time a user issues an SQL query, all the way down to data being read from or written to disk through the OS kernel.

---

### **PostgreSQL I/O Architecture (Text Diagram)**

```
                          ┌──────────────────────────────┐
                          │        SQL Query             │
                          │  (SELECT / INSERT / UPDATE)  │
                          └──────────────┬───────────────┘
                                         │
                                         ▼
                     ┌────────────────────────────────────────┐
                     │         PostgreSQL Executor            │
                     │ (Query Parser, Planner, Executor)      │
                     └──────────────┬─────────────────────────┘
                                    │
                                    ▼
                     ┌────────────────────────────────────────┐
                     │          Access Methods (AM)           │
                     │  Heap, Index (B-Tree, Hash, GiST, etc.)│
                     └──────────────┬─────────────────────────┘
                                    │
                                    ▼
                     ┌────────────────────────────────────────┐
                     │         Buffer Manager (Shared Buffers)│
                     │  - Caches frequently used pages         │
                     │  - Handles read/write to/from disk      │
                     │  - Manages dirty pages (LRU, freelist)  │
                     └──────────────┬─────────────────────────┘
                                    │
                                    │  (If page not in buffer)
                                    ▼
                     ┌────────────────────────────────────────┐
                     │           Storage Manager (SMGR)       │
                     │  - Responsible for physical file I/O   │
                     │  - Interfaces with OS file system APIs  │
                     │  - Handles relation forks:              │
                     │    • main (heap/index)                  │
                     │    • fsm (free space map)               │
                     │    • vm  (visibility map)               │
                     │    • init (unlogged tables)             │
                     └──────────────┬─────────────────────────┘
                                    │
                                    ▼
                     ┌────────────────────────────────────────┐
                     │         OS Kernel / File System         │
                     │  - Manages file descriptors             │
                     │  - Performs page caching (OS cache)     │
                     │  - Schedules block I/O operations       │
                     └──────────────┬─────────────────────────┘
                                    │
                                    ▼
                     ┌────────────────────────────────────────┐
                     │             Disk Storage                │
                     │  - Data files (.dat, heap, index)       │
                     │  - WAL segments (pg_wal/)               │
                     │  - Control files                        │
                     │  - Temporary files                      │
                     └────────────────────────────────────────┘

```

---

### **Write Path Example**

```
SQL INSERT
   ↓
Executor
   ↓
Buffer Manager → Page loaded or created in shared buffers
   ↓
Mark page as "dirty"
   ↓
WAL (Write-Ahead Log) entry created and flushed to pg_wal/
   ↓
Background Writer / Checkpointer writes dirty buffers to disk
   ↓
Storage Manager → OS → Disk
```

---

### **Read Path Example**

```
SQL SELECT
   ↓
Executor requests a tuple
   ↓
Buffer Manager checks if page in shared buffer
   ↓
If not → Storage Manager reads from file via OS
   ↓
Page loaded into shared buffer, tuple returned
```

---

Would you like me to extend this with a **WAL (Write-Ahead Logging) subsystem flow**, showing how the WAL buffer, background writer, and checkpointer coordinate? It completes the I/O picture.


Perfect ✅ — here’s an **extended text-based architecture diagram** showing the **complete PostgreSQL I/O system**, now including the **WAL (Write-Ahead Logging)** subsystem and background processes like **Checkpointer**, **Background Writer**, and **WAL Writer**.

---

## **PostgreSQL I/O Architecture — With WAL Subsystem**

```
                          ┌──────────────────────────────┐
                          │        SQL Query             │
                          │  (SELECT / INSERT / UPDATE)  │
                          └──────────────┬───────────────┘
                                         │
                                         ▼
                     ┌────────────────────────────────────────┐
                     │      Parser / Planner / Executor       │
                     │ (Query execution pipeline)             │
                     └──────────────┬─────────────────────────┘
                                    │
                                    ▼
                     ┌────────────────────────────────────────┐
                     │          Access Methods (AM)           │
                     │  Heap / Index / TOAST / GiST / etc.    │
                     └──────────────┬─────────────────────────┘
                                    │
                                    ▼
                     ┌────────────────────────────────────────┐
                     │        Buffer Manager (Shared Buffers) │
                     │  - Caches data pages                   │
                     │  - Manages dirty pages (LRU, freelist) │
                     │  - Coordinates with WAL subsystem       │
                     └──────────────┬─────────────────────────┘
                                    │
                      ┌─────────────┴───────────────┐
                      │                             │
                      ▼                             ▼
      ┌────────────────────────────────┐   ┌────────────────────────────────┐
      │        WAL Subsystem            │   │        Storage Manager (SMGR) │
      │--------------------------------│   │--------------------------------│
      │ - WAL Buffer (in-memory)       │   │ - Handles file-level I/O       │
      │ - WAL Writer process           │   │ - Reads/writes heap & index     │
      │ - WAL segments (pg_wal/)       │   │ - Manages FSM, VM, INIT forks   │
      │ - Synchronous flush on commit  │   │ - Uses OS system calls          │
      └────────────────────────────────┘   └────────────────────────────────┘
                      │                             │
                      │                             ▼
                      │                ┌────────────────────────────────┐
                      │                │     OS Kernel / File System     │
                      │                │ - Caches file blocks (OS cache) │
                      │                │ - Schedules physical I/O        │
                      │                └──────────────┬─────────────────┘
                      │                               │
                      ▼                               ▼
      ┌──────────────────────────┐       ┌──────────────────────────┐
      │  WAL Files (pg_wal/)     │       │  Data Files (base/...)   │
      │  - Sequential append log │       │  - Heap / Index pages    │
      └──────────────────────────┘       └──────────────────────────┘
```

---

### **💾 Write Path (Detailed Flow)**

```
1️⃣  SQL INSERT / UPDATE / DELETE
     ↓
2️⃣  Executor modifies a tuple → Buffer Manager marks page as "dirty"
     ↓
3️⃣  WAL Record generated → written to WAL buffer
     ↓
4️⃣  On COMMIT:
       → WAL Writer flushes WAL buffer to pg_wal/ on disk (fsync)
       → Ensures durability before confirming commit
     ↓
5️⃣  Background Writer (periodic) writes dirty data pages from shared buffer to disk
     ↓
6️⃣  Checkpointer (periodic) fsyncs all dirty pages to disk and updates checkpoint record in WAL
     ↓
7️⃣  Storage Manager uses OS → Disk write completed
```

---

### **🔍 Read Path (Detailed Flow)**

```
1️⃣  SQL SELECT
     ↓
2️⃣  Executor requests page → Buffer Manager checks shared buffer
     ↓
3️⃣  If page not found → Storage Manager reads from file using OS
     ↓
4️⃣  OS Kernel reads from disk or OS cache
     ↓
5️⃣  Page loaded into shared buffer → Executor reads tuple
```

---

### **⚙️ Background Processes Involved**

| Process               | Role                                                                           |
| --------------------- | ------------------------------------------------------------------------------ |
| **WAL Writer**        | Flushes WAL buffer to disk periodically or on commit.                          |
| **Background Writer** | Writes dirty shared buffers to data files asynchronously.                      |
| **Checkpointer**      | Ensures all dirty pages and WAL up to a point are safely on disk (durability). |
| **Autovacuum**        | Cleans dead tuples, updates visibility map.                                    |
| **Archiver**          | Copies completed WAL segments for backup/recovery.                             |

---

### **🧠 WAL + Buffer Coordination**

```
Before data page is written to disk:
 → Corresponding WAL record *must* be flushed to disk.
This guarantees: "WAL before data" → crash recovery possible.
```

---

Would you like me to add **one more layer showing WAL Replay (Crash Recovery path)** — i.e., how PostgreSQL restores consistency by reapplying WAL during startup?
It completes the lifecycle (write → persist → recover).

