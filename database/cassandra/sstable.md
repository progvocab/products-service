### **🔹 Equivalent of Data Pages in Cassandra**  

In **Cassandra**, the equivalent of **PostgreSQL's data pages** is the **SSTable (Sorted String Table) blocks** stored within **SSTables on disk**. While PostgreSQL organizes data into **fixed-size 8 KB pages**, Cassandra stores data in **immutable SSTables** that use **block-based storage**.

---

## **🔹 How Cassandra Stores Data (Compared to PostgreSQL Pages)**  

| **PostgreSQL** | **Cassandra** | **Description** |
|--------------|------------|----------------|
| **Data Pages (8 KB)** | **SSTable Blocks (variable size)** | PostgreSQL pages store rows, while Cassandra writes data in **blocks inside SSTables**. |
| **Heap Tuples (Rows)** | **Partition Rows** | PostgreSQL tuples are stored in data pages, while Cassandra partitions store rows sorted by clustering keys. |
| **WAL (Write-Ahead Log)** | **Commit Log** | Both databases first write changes to a log for durability before writing to disk. |
| **VACUUM for cleanup** | **Tombstones & Compaction** | PostgreSQL uses **VACUUM** to clean dead tuples; Cassandra relies on **Tombstones + Compaction**. |

---

## **🔹 Cassandra's SSTable Block Storage**  
When data is written to **Cassandra**, it follows this process:

1️⃣ **Memtable (In-Memory Write)** → Writes go to an in-memory structure.  
2️⃣ **Commit Log (Durability)** → Ensures crash recovery.  
3️⃣ **SSTable (On-Disk Storage in Blocks)** → Periodically, the **Memtable is flushed to an SSTable** stored in **blocks**.  
4️⃣ **Compaction** → Old SSTables are merged, and **deleted data (tombstones) are purged**.

---

## **🔹 Data Layout in SSTables (Similar to Pages in PostgreSQL)**  
Each **SSTable contains multiple blocks**, and each block consists of:

| **SSTable Component** | **Description** |
|-----------------|----------------|
| **Primary Index** | Helps locate data in SSTables efficiently. |
| **Partition Index** | Stores partition keys and offsets for quick lookups. |
| **Bloom Filter** | Reduces unnecessary disk reads. |
| **Data Blocks** | Actual rows stored in **compressed blocks**. |

---

## **🔹 Key Differences Between PostgreSQL Pages & Cassandra Blocks**  

| **Feature** | **PostgreSQL (Pages)** | **Cassandra (SSTable Blocks)** |
|------------|------------------|------------------|
| **Data Organization** | Fixed **8 KB pages** | Variable **block sizes** |
| **Write Behavior** | Overwrites existing pages | **Immutable SSTables (append-only)** |
| **Storage Cleanup** | Uses `VACUUM` | Uses **Tombstones + Compaction** |
| **Read Efficiency** | B-Trees & indexes optimize lookups | **Partition-based** row storage |
| **Concurrency Model** | MVCC (Multi-Version Concurrency Control) | **Eventual Consistency** |

---

## **🔹 Summary**
- **Cassandra does not use fixed-size pages like PostgreSQL** but instead stores data in **SSTable blocks**.
- **PostgreSQL modifies pages in place**, while **Cassandra writes immutable SSTables** and relies on **compaction** to merge data.
- **Cassandra optimizes for write-heavy workloads**, whereas **PostgreSQL optimizes for transactional consistency**.

Would you like a **deep dive into SSTable structure or read/write performance comparison**? 🚀