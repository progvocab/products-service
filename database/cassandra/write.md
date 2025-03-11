### **How Cassandra Writes Data to the File System**  

When a **new record** is inserted into **Apache Cassandra (a wide-column store)**, it follows a **Log-Structured Merge (LSM) tree** approach to handle writes efficiently. The write path ensures **high throughput** and **durability** while keeping reads performant.

---

## **1. Cassandra Write Path (Step-by-Step)**  
When a client inserts a new record:  

1. **Write to Commit Log** (for durability)  
2. **Write to Memtable (in-memory table)**  
3. **Memtable Flushes to SSTables (immutable on-disk storage)**  
4. **Compaction Merges SSTables for Optimization**  

---

## **2. Step-by-Step Breakdown of Write Operations**
### **Step 1: Write to Commit Log (WAL - Write-Ahead Log)**
- Every write is **first stored in the Commit Log**, a sequential **append-only file** on disk.
- Ensures **durability**—if a node crashes, data can be recovered from the log.

✅ **Fast (sequential writes to disk)**  
✅ **Durable (prevents data loss in case of failure)**  

💡 **Example Commit Log Entry**
```
{timestamp: 171020241235, key: 101, column: name, value: "Alice"}
```

---

### **Step 2: Write to Memtable (In-Memory Structure)**
- Data is written **asynchronously** to the **Memtable**.
- Memtable is **sorted by partition key** for efficient reads.

💡 **Example Memtable (Before Flush)**
| Partition Key | Column | Value  |
|--------------|--------|--------|
| 101          | name   | Alice  |
| 102          | age    | 30     |

---

### **Step 3: Flushing Memtable to SSTables (On-Disk Storage)**
- When the **Memtable is full**, it is **flushed** to an **SSTable (Sorted String Table)** on disk.
- The SSTable is **immutable** (never updated once written).
- **Index files** are created to speed up lookups.

💡 **Example SSTable File**
```
data-0001-101.db  (Key: 101 → Name: Alice)
data-0001-102.db  (Key: 102 → Age: 30)
```

✅ **Fast sequential disk writes (append-only SSTables)**  
✅ **Optimized for read performance (sorted data, indexed)**  

---

### **Step 4: Compaction (Merging SSTables)**
- Over time, multiple SSTables exist.
- **Compaction merges older SSTables**, **removing tombstones (deleted data)** and **reducing read latency**.

✅ **Improves read efficiency**  
✅ **Cleans up deleted data (garbage collection)**  

💡 **Before Compaction:**
```
SSTable1: {101 → Alice}, {102 → 30}
SSTable2: {101 → Alice Johnson}  (newer update)
```
💡 **After Compaction:**
```
SSTable3: {101 → Alice Johnson}, {102 → 30}
```
Now, Cassandra reads from **fewer SSTables**, improving performance.

---

## **3. Summary: How Cassandra Handles Writes**
| Step | Action | Purpose |
|------|--------|---------|
| **1** | Write to **Commit Log** | Ensures durability (WAL) |
| **2** | Write to **Memtable** (RAM) | Fast writes, sorted by key |
| **3** | Flush to **SSTables** | Persistent storage (immutable) |
| **4** | **Compaction** | Merges old data, removes tombstones |

---

### **Why This Approach?**
- **High Write Throughput** → **Sequential writes** (Commit Log, SSTables).  
- **Read Optimization** → **Sorted data, indexed SSTables**.  
- **Scalability** → Designed for **distributed storage**.  

Would you like a deep dive into **compaction strategies or read paths** in Cassandra?

### **Cassandra Read & Compaction Strategies (Deep Dive)**  

Now that we've covered **how Cassandra writes data**, let's dive into:  
1. **Read Path** – How Cassandra retrieves data efficiently.  
2. **Compaction Strategies** – How Cassandra optimizes storage and read performance.  

---

## **1. Cassandra Read Path (How Reads Work)**  

When a client queries Cassandra, the system must efficiently fetch the requested data from multiple sources:

### **Step 1: Check Row Cache (Optional)**
- If **row caching is enabled**, Cassandra first checks if the data is **cached in memory**.
- **Speeds up repeated queries**.

✅ **Fastest** if data is found in cache.  
❌ **Not always available** (depends on configuration).  

---

### **Step 2: Check Memtable (In-Memory Data)**
- If the data **is not cached**, Cassandra checks the **Memtable (RAM)**.  
- Memtable is a sorted structure that contains **recent writes** that haven't yet been flushed to disk.

✅ **Fast retrieval from RAM**  
❌ **Only contains recent data**  

---

### **Step 3: Check Bloom Filter (Quick SSTable Lookup)**
- If the data **is not in Memtable**, Cassandra **uses a Bloom filter** to check which SSTables **might** contain the requested partition.
- **Bloom filters** are probabilistic data structures that **quickly eliminate SSTables that don’t have the data**.

✅ **Avoids unnecessary SSTable reads**  
❌ **May have false positives (but no false negatives)**  

---

### **Step 4: Check SSTables (Disk Storage)**
- If needed, Cassandra loads data from **SSTables on disk**.
- **SSTable components:**  
  - **Primary index** – Quick lookup by partition key.  
  - **Partition summary** – Points to partition locations in SSTable.  
  - **Partition index** – Stores exact locations of partitions.  
  - **Data file** – Stores actual column values.  

✅ **Efficient because SSTables are sorted**  
❌ **Might require checking multiple SSTables (if not compacted)**  

💡 **Example Query**
```cql
SELECT name FROM users WHERE user_id = 101;
```
🔍 **Lookup Flow:**
1. **Check Memtable** → If not found  
2. **Check Bloom Filter** → Identify relevant SSTables  
3. **Read from SSTables** → Merge and return data  

---

### **Step 5: Merge SSTable Data (If Needed)**
- If the same key exists in **multiple SSTables**, Cassandra merges results **on the fly**.
- **Newest version wins** (based on timestamp).

---

## **2. Compaction Strategies in Cassandra**  
Cassandra's **compaction process** periodically merges SSTables to:  
✅ **Reduce read latency**  
✅ **Remove tombstones (deleted data)**  
✅ **Reduce storage fragmentation**  

### **Types of Compaction Strategies**
1. **Size-Tiered Compaction (Default)**
2. **Leveled Compaction**
3. **Time-Window Compaction (For Time-Series Data)**

---

### **1. Size-Tiered Compaction (STCS - Default)**
- **Best for write-heavy workloads**.
- Merges **smaller SSTables into larger ones** when a certain number of SSTables accumulate.

💡 **Example:**
```
SSTable1 (100MB) + SSTable2 (100MB) → SSTable3 (200MB)
```

✅ **Efficient for bulk writes**  
❌ **Reads may need to check multiple SSTables**  

---

### **2. Leveled Compaction (LCS)**
- **Best for read-heavy workloads**.
- Uses **fixed-size levels** (L1, L2, L3, etc.), ensuring that **data is evenly distributed**.

💡 **Example:**
```
L1: 10MB SSTables  
L2: 100MB SSTables (Merged from L1)  
L3: 1GB SSTables (Merged from L2)
```

✅ **Reduces read latency (fewer SSTables to check)**  
❌ **More disk I/O due to frequent compactions**  

**Enable LCS:**
```cql
ALTER TABLE users WITH compaction = {'class': 'LeveledCompactionStrategy'};
```

---

### **3. Time-Window Compaction (TWCS - Best for Time-Series Data)**
- Optimized for **time-series workloads** (e.g., logs, IoT data).
- Compacts SSTables **within time windows** (e.g., daily or hourly).

✅ **Prevents old data from being rewritten**  
✅ **Best for time-ordered inserts**  

**Enable TWCS:**
```cql
ALTER TABLE sensor_data WITH compaction = {'class': 'TimeWindowCompactionStrategy', 'compaction_window_size': '1', 'compaction_window_unit': 'DAYS'};
```

---

## **3. Summary: Read & Compaction**
| Step | Action | Purpose |
|------|--------|---------|
| **1** | Check **Row Cache** | Avoids query processing (fastest) |
| **2** | Check **Memtable** | Retrieves recently written data |
| **3** | Check **Bloom Filter** | Identifies SSTables with relevant data |
| **4** | Read from **SSTables** | Fetches older data (disk read) |
| **5** | **Compaction** | Merges SSTables, removes tombstones |

| Compaction Strategy | Best Use Case |
|----------------------|--------------|
| **Size-Tiered (STCS)** | Write-heavy workloads |
| **Leveled (LCS)** | Read-heavy workloads |
| **Time-Window (TWCS)** | Time-series data |

---

### **Final Thoughts**
- **Cassandra is optimized for writes**, but **reads can slow down if there are too many SSTables**.  
- **Choosing the right compaction strategy** improves read performance.  
- **Bloom filters and partition indexes** help minimize disk access.  

Would you like an example of **real-world Cassandra query optimization** or **benchmarking different compaction strategies**?