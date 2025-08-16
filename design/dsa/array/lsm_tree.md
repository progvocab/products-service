## **Log-Structured Merge Tree (LSM Tree) - Explained with Code Examples**

### **1️⃣ What is an LSM Tree?**
An **LSM Tree (Log-Structured Merge Tree)** is a data structure optimized for **high write throughput**. It is commonly used in databases like **Cassandra, LevelDB, RocksDB, and ScyllaDB**.

### **Key Features:**
✅ **Fast Writes**: Data is written sequentially in memory.  
✅ **Efficient Merging**: Data is periodically merged in the background.  
✅ **Indexing**: Uses **SSTables (Sorted String Tables)** instead of B-Trees.  
✅ **Compaction**: Old data is periodically merged and deleted.  

---

## **2️⃣ How LSM Tree Works**
1. **Writes go to a Memtable (in-memory sorted structure).**  
2. **When Memtable is full, it is flushed to disk as an immutable SSTable.**  
3. **Background merge process (Compaction) merges multiple SSTables into larger sorted files.**  
4. **Reads involve checking Memtable first, then SSTables (using Bloom filters and indexes).**  

---

## **3️⃣ Implementing LSM Tree in Python**
We will **simulate** an LSM Tree with:
- **Memtable (in-memory store)**
- **SSTables (sorted on-disk segments)**
- **Compaction (merging SSTables)**

---

### **Step 1: Define an LSM Tree Class**
```python
import os
import json
import heapq

class LSMTree:
    def __init__(self, memtable_limit=3):
        self.memtable = {}  # In-memory store
        self.memtable_limit = memtable_limit  # Max records before flushing
        self.sstables = []  # List of SSTables (on-disk files)
        self.sstable_count = 0  # SSTable file counter

    def put(self, key, value):
        """Insert data into the Memtable."""
        self.memtable[key] = value

        # Flush Memtable to SSTable when limit is reached
        if len(self.memtable) >= self.memtable_limit:
            self.flush_to_sstable()

    def flush_to_sstable(self):
        """Flush Memtable to disk as an SSTable."""
        if not self.memtable:
            return
        
        # Sort Memtable before writing
        sorted_data = sorted(self.memtable.items())

        # Write to a new SSTable file
        filename = f"sstable_{self.sstable_count}.json"
        with open(filename, "w") as f:
            json.dump(sorted_data, f)

        self.sstables.append(filename)
        self.sstable_count += 1
        self.memtable.clear()  # Clear Memtable after flushing

    def get(self, key):
        """Retrieve data, checking Memtable first, then SSTables."""
        if key in self.memtable:
            return self.memtable[key]
        
        # Search SSTables (from newest to oldest)
        for sstable in reversed(self.sstables):
            with open(sstable, "r") as f:
                data = json.load(f)
                for k, v in data:
                    if k == key:
                        return v
        return None  # Key not found

    def compact(self):
        """Merge multiple SSTables into a single sorted SSTable."""
        merged_data = []

        # Read all SSTables and merge sorted data
        for sstable in self.sstables:
            with open(sstable, "r") as f:
                data = json.load(f)
                merged_data.extend(data)

        # Sort merged data and remove duplicates (keep latest value)
        merged_data = sorted(dict(merged_data).items())

        # Write to a new compacted SSTable
        compacted_file = f"sstable_compacted.json"
        with open(compacted_file, "w") as f:
            json.dump(merged_data, f)

        # Remove old SSTables
        for sstable in self.sstables:
            os.remove(sstable)

        self.sstables = [compacted_file]  # Keep only the compacted SSTable
```

---

### **Step 2: Insert and Retrieve Data**
```python
# Create an LSM Tree instance
lsm = LSMTree(memtable_limit=3)

# Insert data (triggers flush when limit is reached)
lsm.put("apple", 100)
lsm.put("banana", 200)
lsm.put("cherry", 300)  # This will trigger a flush
lsm.put("date", 400)
lsm.put("elderberry", 500)
lsm.put("fig", 600)  # Another flush happens

print("Retrieving values:")
print("banana:", lsm.get("banana"))  # 200
print("fig:", lsm.get("fig"))  # 600
print("grape:", lsm.get("grape"))  # None (not found)
```

#### **Output:**
```
Retrieving values:
banana: 200
fig: 600
grape: None
```

---

### **Step 3: Perform Compaction**
```python
print("Compacting SSTables...")
lsm.compact()

print("Retrieving after compaction:")
print("banana:", lsm.get("banana"))  # 200
print("fig:", lsm.get("fig"))  # 600
```

#### **Compaction Process**
- Merges multiple small SSTables into **one sorted file**.
- Keeps **only the latest version** of each key.
- Deletes **old SSTables** to free up space.

---

## **4️⃣ Summary**
✅ **LSM Trees use in-memory writes (Memtable) for speed**.  
✅ **SSTables (sorted files) store flushed data**.  
✅ **Compaction merges SSTables to improve read performance**.  
✅ **Efficient for databases with high write workloads** (like Cassandra & RocksDB).  

---

### **Would you like an optimized version with Bloom Filters for faster reads? 🚀**