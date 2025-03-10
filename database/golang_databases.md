There are a few databases written in **Go (Golang)**, but Go is primarily used for building **database clients, drivers, and distributed systems** rather than full database engines. However, some Go-based databases exist, particularly for **embedded use, distributed systems, and NoSQL applications**.  

---

## **🔹 Databases Written in Go**
### **1️⃣ BoltDB**  
✔ **Type**: Key-Value Store (Embedded, NoSQL)  
✔ **Use Case**: Lightweight, single-file database (like SQLite but for key-value)  
✔ **Features**:  
   - **Pure Go** implementation  
   - **ACID transactions**  
   - **Low-latency reads & writes**  
   - Persistent key-value storage  
✔ **Example Use Case**: Used in **Docker, Kubernetes, and etcd**  
✔ **GitHub**: [github.com/boltdb/bolt](https://github.com/boltdb/bolt)  

#### **Usage Example (BoltDB)**
```go
package main

import (
	"fmt"
	"go.etcd.io/bbolt"
)

func main() {
	db, _ := bbolt.Open("my.db", 0600, nil)
	defer db.Close()

	db.Update(func(tx *bbolt.Tx) error {
		b, _ := tx.CreateBucketIfNotExists([]byte("MyBucket"))
		return b.Put([]byte("key"), []byte("value"))
	})

	db.View(func(tx *bbolt.Tx) error {
		b := tx.Bucket([]byte("MyBucket"))
		fmt.Println(string(b.Get([]byte("key")))) // Output: value
		return nil
	})
}
```

---

### **2️⃣ BadgerDB**  
✔ **Type**: Key-Value Store (NoSQL, Embedded, High-Performance)  
✔ **Use Case**: Fast, persistent key-value store (like RocksDB)  
✔ **Features**:  
   - **Optimized for SSDs**  
   - **Lock-free architecture** for fast reads/writes  
   - **No background garbage collection** (unlike BoltDB)  
✔ **Example Use Case**: Used in **Dgraph, Jaeger, IPFS**  
✔ **GitHub**: [github.com/dgraph-io/badger](https://github.com/dgraph-io/badger)  

#### **Usage Example (BadgerDB)**
```go
package main

import (
	"fmt"
	"log"

	"github.com/dgraph-io/badger/v4"
)

func main() {
	opts := badger.DefaultOptions("badgerdb")
	db, err := badger.Open(opts)
	if err != nil {
		log.Fatal(err)
	}
	defer db.Close()

	err = db.Update(func(txn *badger.Txn) error {
		return txn.Set([]byte("key"), []byte("value"))
	})

	err = db.View(func(txn *badger.Txn) error {
		item, err := txn.Get([]byte("key"))
		if err != nil {
			return err
		}
		val, _ := item.ValueCopy(nil)
		fmt.Println(string(val)) // Output: value
		return nil
	})
}
```

---

### **3️⃣ dgraph**  
✔ **Type**: Distributed Graph Database (NoSQL)  
✔ **Use Case**: Scalable, highly available graph database  
✔ **Features**:  
   - **GraphQL & gRPC APIs**  
   - **Distributed & scalable**  
   - **High performance with SSD optimization**  
✔ **Example Use Case**: Social networks, recommendation engines, fraud detection  
✔ **GitHub**: [github.com/dgraph-io/dgraph](https://github.com/dgraph-io/dgraph)  

---

### **4️⃣ Tiedot**  
✔ **Type**: Document-Oriented NoSQL Database (JSON-based)  
✔ **Use Case**: Lightweight NoSQL storage for structured data  
✔ **Features**:  
   - **Schema-less JSON storage**  
   - **RESTful API for CRUD operations**  
   - **Written entirely in Go**  
✔ **Example Use Case**: JSON-based document storage for **small projects**  
✔ **GitHub**: [github.com/HouzuoGuo/tiedot](https://github.com/HouzuoGuo/tiedot)  

---

### **5️⃣ Cayley**  
✔ **Type**: Graph Database  
✔ **Use Case**: Relationship-heavy data storage  
✔ **Features**:  
   - **GraphQL & Gremlin support**  
   - **Works with various backends (BoltDB, PostgreSQL, etc.)**  
✔ **Example Use Case**: Social networks, recommendation engines  
✔ **GitHub**: [github.com/cayleygraph/cayley](https://github.com/cayleygraph/cayley)  

---

### **6️⃣ Pilosa**  
✔ **Type**: Bitmap Index Database (Analytical Queries)  
✔ **Use Case**: Fast queries over **large datasets**  
✔ **Features**:  
   - **Optimized for Boolean operations**  
   - **High-performance indexing**  
✔ **Example Use Case**: **Real-time analytics & log analysis**  
✔ **GitHub**: [github.com/pilosa/pilosa](https://github.com/pilosa/pilosa)  

---

### **7️⃣ VictoriaMetrics**  
✔ **Type**: Time-Series Database (TSDB)  
✔ **Use Case**: **High-performance time-series storage**  
✔ **Features**:  
   - **Optimized for large-scale metrics**  
   - **Fast queries & efficient storage**  
✔ **Example Use Case**: Used in **Prometheus-compatible monitoring**  
✔ **GitHub**: [github.com/VictoriaMetrics/VictoriaMetrics](https://github.com/VictoriaMetrics/VictoriaMetrics)  

---

## **🔹 Go-Based Distributed Databases**
Though not fully written in Go, these databases use Go extensively:

| **Database** | **Written In** | **Type** | **GitHub** |
|-------------|---------------|----------|------------|
| **etcd** | Go | Key-Value Store | [github.com/etcd-io/etcd](https://github.com/etcd-io/etcd) |
| **CockroachDB** | Go & C++ | SQL, Distributed | [github.com/cockroachdb/cockroach](https://github.com/cockroachdb/cockroach) |
| **TiDB** | Go & Rust | NewSQL (MySQL Compatible) | [github.com/pingcap/tidb](https://github.com/pingcap/tidb) |
| **InfluxDB** | Go | Time-Series Database | [github.com/influxdata/influxdb](https://github.com/influxdata/influxdb) |

---

## **🚀 Summary: Which Go Database Should You Use?**
| **Database** | **Best For** |
|-------------|-------------|
| **BoltDB** | Simple key-value storage (lightweight apps) |
| **BadgerDB** | High-performance key-value storage |
| **dgraph** | Graph-based data (relationships, recommendations) |
| **Tiedot** | JSON document storage |
| **Cayley** | Graph databases with multiple backends |
| **Pilosa** | Fast bitmap-based analytical queries |
| **VictoriaMetrics** | Time-series data (monitoring, analytics) |
| **etcd** | Distributed key-value storage (config, service discovery) |
| **CockroachDB** | Distributed SQL (high availability) |
| **TiDB** | MySQL-compatible distributed SQL |

Go is mostly used for **high-performance distributed systems**, and many large-scale databases (etcd, CockroachDB, TiDB, InfluxDB) are written in Go.

Would you like recommendations based on a specific **use case**?