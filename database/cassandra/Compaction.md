### **🔹 Cassandra Compaction: How It Works & Why It Matters**  

Since Cassandra uses **Log-Structured Merge (LSM) Trees** instead of updating data in place (like PostgreSQL), it needs **compaction** to optimize storage and improve read performance.  

---

## **1️⃣ Why is Compaction Needed in Cassandra?**  
- Cassandra **writes new data to immutable SSTables**, without updating old ones.  
- This leads to **multiple SSTables** containing different versions of the same data.  
- During reads, Cassandra has to **check multiple SSTables**, making queries slower.  
- **Compaction merges and removes old SSTables**, keeping only the latest version.  

---

## **2️⃣ How Cassandra Compaction Works**
### **🔹 1. Data Flow Before Compaction**
1️⃣ **Writes go to a Commit Log** (for durability).  
2️⃣ **Data is stored in a Memtable** (in-memory storage).  
3️⃣ **Memtable is flushed to disk as a new SSTable** (sorted & immutable).  
4️⃣ **Multiple SSTables accumulate**, slowing down reads.  
5️⃣ **Compaction merges SSTables, removing old versions and deleted data.**  

---

## **3️⃣ Types of Compaction in Cassandra**
### **🔹 1. Size-Tiered Compaction (Default)**
✔ **Best for write-heavy workloads**  
✔ Merges small SSTables into larger ones over time  
✔ Efficient for bulk writes but can cause **high read amplification**  
   
📌 **Example:**  
- 4 small SSTables (100MB each) → Compacted into 1 larger SSTable (400MB)  
- Fewer SSTables mean **faster reads**  

---

### **🔹 2. Leveled Compaction (LCS)**
✔ **Best for read-heavy workloads**  
✔ Organizes SSTables into levels (`L0 → L1 → L2 ...`)  
✔ Only merges overlapping SSTables, reducing read amplification  

📌 **Example:**  
- SSTables are **small & more evenly distributed** across levels  
- Faster read access because **only relevant SSTables are checked**  

---

### **🔹 3. Time-Window Compaction (TWCS)**
✔ **Best for time-series data (IoT, logs, etc.)**  
✔ Groups SSTables by **time ranges** (e.g., daily, hourly)  
✔ Prevents unnecessary merges of old data  

📌 **Example:**  
- Data from **March 1** stays in an SSTable for **March 1**  
- Older SSTables remain untouched, improving performance  

---

## **4️⃣ How Compaction Affects Performance**
🔹 **Pros:**  
✅ **Faster Reads**: Reduces the number of SSTables Cassandra needs to check.  
✅ **Less Disk Space Usage**: Removes deleted data & old versions.  
✅ **Better Query Performance**: Optimizes index lookups.  

🔹 **Cons:**  
❌ **High Disk I/O**: Compaction can be **CPU- and IO-intensive**.  
❌ **Temporary Space Usage**: Needs extra disk space during merging.  
❌ **Increased Latency During Compaction**: Queries may slow down.  

---

## **5️⃣ How to Configure Compaction in Cassandra**
You can specify the compaction strategy when creating a table:  
```sql
CREATE TABLE employee_metrics (
    employee_id UUID,
    date DATE,
    total_hours DOUBLE,
    PRIMARY KEY (employee_id, date)
) WITH compaction = { 'class': 'LeveledCompactionStrategy' };
```
Or update an existing table:  
```sql
ALTER TABLE employee_metrics 
WITH compaction = { 'class': 'TimeWindowCompactionStrategy', 'compaction_window_size': '7', 'compaction_window_unit': 'DAYS' };
```

---

## **6️⃣ Summary: Which Compaction Strategy to Use?**
| **Compaction Type** | **Best for** | **Pros** | **Cons** |
|---------------------|-------------|----------|----------|
| **Size-Tiered (STCS)** | Write-heavy workloads | Good for bulk inserts | High read amplification |
| **Leveled (LCS)** | Read-heavy workloads | Faster queries, fewer SSTables | More disk writes |
| **Time-Window (TWCS)** | Time-series data | Efficient for logs, IoT data | Not suitable for frequent updates |

Would you like a **real-world example with performance tuning tips?** 🚀