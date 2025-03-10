Several major databases are written in **C++** due to its **high performance, memory efficiency, and control over system resources**. Here are some of the most notable ones:  

---

## **🔹 Major Databases Written in C++**

### **1️⃣ MySQL**  
- **Type**: Relational Database (RDBMS)  
- **Use Cases**: Web applications, enterprise applications, transactional systems  
- **Why C++?**: High performance, efficient memory management, and optimized query execution  
- **Notable Users**: Facebook, Twitter, YouTube  
- **Website**: [https://www.mysql.com/](https://www.mysql.com/)

---

### **2️⃣ PostgreSQL (Partially C++)**  
- **Type**: Relational Database (RDBMS)  
- **Use Cases**: Data warehousing, analytics, transactional applications  
- **Why C++?**: Some parts of PostgreSQL, like its query planner, are implemented in C++, though the core is written in **C**.  
- **Notable Users**: Apple, Reddit, Instagram  
- **Website**: [https://www.postgresql.org/](https://www.postgresql.org/)

---

### **3️⃣ MongoDB**  
- **Type**: NoSQL Document Database  
- **Use Cases**: Big Data, real-time analytics, IoT applications  
- **Why C++?**: C++ offers **high-speed data processing**, making MongoDB suitable for large-scale, high-traffic applications.  
- **Notable Users**: eBay, Adobe, Uber  
- **Website**: [https://www.mongodb.com/](https://www.mongodb.com/)

---

### **4️⃣ RocksDB**  
- **Type**: Embedded Key-Value Store  
- **Use Cases**: High-performance caching, logging systems, real-time analytics  
- **Why C++?**: Optimized for SSD storage, low-latency access, and multi-threaded workloads  
- **Notable Users**: Facebook, LinkedIn, Kafka  
- **Website**: [https://rocksdb.org/](https://rocksdb.org/)

---

### **5️⃣ LevelDB**  
- **Type**: Embedded Key-Value Store  
- **Use Cases**: Browser storage (used in Chrome), mobile databases, lightweight applications  
- **Why C++?**: Fast key-value storage with a **simple API and minimal overhead**  
- **Notable Users**: Google Chrome, Bitcoin Core  
- **Website**: [https://github.com/google/leveldb](https://github.com/google/leveldb)

---

### **6️⃣ ClickHouse**  
- **Type**: Columnar OLAP Database  
- **Use Cases**: Real-time analytics, big data processing  
- **Why C++?**: Handles **millions of rows per second**, optimized for analytical workloads  
- **Notable Users**: Cloudflare, Uber, Spotify  
- **Website**: [https://clickhouse.com/](https://clickhouse.com/)

---

### **7️⃣ Redis (Some Parts in C++)**  
- **Type**: In-Memory Key-Value Store  
- **Use Cases**: Caching, real-time leaderboards, session storage  
- **Why C++?**: While Redis is mostly written in **C**, some modules use C++ for advanced features.  
- **Notable Users**: Twitter, GitHub, Snapchat  
- **Website**: [https://redis.io/](https://redis.io/)

---

## **🔹 Why Are Many Databases Written in C++?**
- **High performance**: C++ provides low-level control over memory and CPU usage.  
- **Concurrency support**: Essential for handling multiple database transactions efficiently.  
- **Efficient resource management**: C++ avoids garbage collection overhead seen in languages like Java.  
- **Portability**: Works across multiple operating systems without performance trade-offs.  

---

### **🚀 Summary**
| **Database**   | **Type**                     | **Notable Users** |
|---------------|----------------------------|-------------------|
| **MySQL**     | Relational (SQL)           | Facebook, Twitter, YouTube |
| **PostgreSQL** | Relational (SQL) (partially C++) | Apple, Reddit, Instagram |
| **MongoDB**   | NoSQL (Document Store)      | eBay, Uber, Adobe |
| **RocksDB**   | Embedded Key-Value Store    | Facebook, LinkedIn |
| **LevelDB**   | Embedded Key-Value Store    | Google Chrome, Bitcoin Core |
| **ClickHouse** | Columnar OLAP DB           | Cloudflare, Uber |
| **Redis**     | In-Memory Key-Value Store   | Twitter, Snapchat |

Would you like a comparison between these databases based on **performance or scalability**?