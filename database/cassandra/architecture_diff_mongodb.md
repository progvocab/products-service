### **MongoDB vs. Cassandra: Architectural Differences**  

Both **MongoDB** and **Cassandra** are NoSQL databases, but they have **fundamentally different architectures** that impact their performance, scalability, and use cases.  

---

## **1️⃣ High-Level Comparison**  

| Feature         | **MongoDB** (Document Store) | **Cassandra** (Columnar Store) |
|---------------|------------------------|------------------------|
| **Data Model** | JSON-like documents (BSON) | Wide-column store (Keyspace → Tables → Partitions) |
| **Query Language** | MongoDB Query Language (MQL) | Cassandra Query Language (CQL) |
| **Architecture** | Master-Slave (Replica Sets, Sharding) | Masterless, Peer-to-Peer (All nodes are equal) |
| **Consistency Model** | Strong consistency by default | Tunable consistency (CAP Theorem-based) |
| **Scalability** | Scales well with **Sharding** | Linearly scalable with **peer-to-peer replication** |
| **Write Performance** | Fast for single-document writes | Optimized for high write throughput |
| **Read Performance** | Faster for single-record lookups | Optimized for sequential reads |
| **Failure Handling** | Automatic failover (Replica sets) | Highly available (No single point of failure) |
| **Best for** | **Dynamic, flexible schema**, analytics | **High write throughput, availability** |

---

## **2️⃣ Architectural Breakdown**
### **🔹 MongoDB Architecture**
MongoDB follows a **Master-Slave** model with optional **Sharding** for scalability.  

#### **📌 Key Components**
1. **Replica Sets** (High Availability)  
   - A **Primary node** handles all writes.  
   - **Secondary nodes** replicate data and can serve read queries.  
   - Automatic **failover** if the primary node goes down.  

2. **Sharding** (Horizontal Scalability)  
   - Distributes data across multiple **shards (nodes)**.  
   - Each shard is a replica set.  
   - **Mongos router** helps applications query the correct shard.  

3. **BSON Documents** (Flexible Schema)  
   - Data is stored as **JSON-like** objects.  
   - Indexing on fields improves query performance.  

#### **🔹 Example: MongoDB Replica Set**
```
+------------+    +------------+    +------------+
|  PRIMARY   | -> |  SECONDARY  | -> |  SECONDARY  |
|  (Writes)  |    |  (Replicas) |    |  (Replicas) |
+------------+    +------------+    +------------+
```

✅ **Best for:** Applications requiring **flexible schema, ad-hoc queries, and analytics**.

---

### **🔹 Cassandra Architecture**
Cassandra follows a **Masterless Peer-to-Peer** model designed for **high availability and scalability**.

#### **📌 Key Components**
1. **Peer-to-Peer Replication**  
   - Every node is **equal** (no master/slave).  
   - Data is **partitioned** across nodes using a **consistent hashing ring**.  
   - Nodes **replicate** data to each other.  

2. **Data Storage Model (Columnar)**
   - Data is stored in **Keyspaces** → **Tables** → **Partitions** → **Columns**.
   - Efficient for **large-scale writes and sequential reads**.  

3. **Tunable Consistency**
   - Allows choosing between **strong or eventual consistency**.
   - Example: `QUORUM`, `ONE`, `ALL` read/write consistency levels.  

#### **🔹 Example: Cassandra Ring Architecture**
```
+------------+    +------------+    +------------+
|   Node 1   | -> |   Node 2   | -> |   Node 3   |
|  (Replica) |    |  (Replica) |    |  (Replica) |
+------------+    +------------+    +------------+
        |                |                |
        +----> Writes distributed <----+
```

✅ **Best for:** **Write-heavy applications, IoT, time-series data**.

---

## **3️⃣ Key Architectural Differences**
| **Feature** | **MongoDB** | **Cassandra** |
|------------|------------|------------|
| **Architecture** | Master-Slave (Primary/Secondary) | Peer-to-Peer, Masterless |
| **Data Model** | Document-based (BSON) | Wide-column store (Row-oriented) |
| **Consistency** | Strong (default) | Tunable (Choose consistency level) |
| **Scalability** | Sharding required for horizontal scaling | Automatically scales linearly |
| **Read Performance** | Faster for single-record lookups | Optimized for sequential reads |
| **Write Performance** | Slower under high load | Very fast for distributed writes |
| **Failure Recovery** | Automatic failover | No single point of failure |
| **Best Use Case** | Analytics, flexible schema, real-time queries | High write workloads, time-series data |

---

## **4️⃣ When to Choose MongoDB vs Cassandra?**
| **Use Case** | **MongoDB** | **Cassandra** |
|-------------|------------|------------|
| **Real-Time Analytics** | ✅ | ❌ |
| **User Profiles, JSON API** | ✅ | ❌ |
| **High Write Volume (Logs, IoT, Time-series)** | ❌ | ✅ |
| **E-Commerce Product Catalog** | ✅ | ❌ |
| **Fraud Detection (Low Latency, High Availability)** | ❌ | ✅ |

---

## **🚀 Conclusion**
- **MongoDB** → Best for **dynamic, document-based applications** with complex queries.  
- **Cassandra** → Best for **high availability, write-intensive workloads, and distributed systems**.  

Would you like **code examples** for querying MongoDB and Cassandra? 🚀