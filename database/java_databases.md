There are several databases written in **Java**. These databases are typically designed to be **portable**, **embeddable**, or **scalable** for Java-based applications.  

---

## **🔹 List of Databases Written in Java**
### **1️⃣ H2 Database**  
✔ **Type**: Relational (SQL)  
✔ **Use Case**: Embedded or standalone DB for testing & lightweight applications  
✔ **Features**:  
   - Supports **in-memory** and **persistent storage**  
   - **Lightweight & fast**  
   - Supports **PostgreSQL compatibility mode**  
   - JDBC API support  
✔ **Example Use Case**: Used in **Spring Boot** applications for testing  
✔ **Website**: [www.h2database.com](https://www.h2database.com/)  

#### **Usage Example in Java (H2)**
```java
Connection conn = DriverManager.getConnection("jdbc:h2:mem:testdb", "sa", "");
Statement stmt = conn.createStatement();
stmt.execute("CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255))");
```

---

### **2️⃣ Apache Derby (JavaDB)**  
✔ **Type**: Relational (SQL)  
✔ **Use Case**: Embedded database for Java applications  
✔ **Features**:  
   - **Pure Java** implementation  
   - **Embedded and network server mode**  
   - Supports **JDBC & SQL**  
✔ **Example Use Case**: Used in **Java SE** as `JavaDB`  
✔ **Website**: [db.apache.org/derby/](https://db.apache.org/derby/)  

#### **Usage Example in Java (Derby)**
```java
Connection conn = DriverManager.getConnection("jdbc:derby:memory:myDB;create=true");
Statement stmt = conn.createStatement();
stmt.execute("CREATE TABLE employees (id INT, name VARCHAR(100))");
```

---

### **3️⃣ HyperSQL (HSQLDB)**  
✔ **Type**: Relational (SQL)  
✔ **Use Case**: Lightweight, fast database for Java applications  
✔ **Features**:  
   - Supports **in-memory** and **file-based** storage  
   - Supports **transactions and ACID compliance**  
   - Compatible with **JDBC, ODBC, and SQL:2016 standard**  
✔ **Example Use Case**: Used in **JUnit tests** and Java desktop applications  
✔ **Website**: [hsqldb.org](http://hsqldb.org/)  

---

### **4️⃣ OrientDB**  
✔ **Type**: Multi-model (Graph + Document + Key-Value + Object-Oriented)  
✔ **Use Case**: Graph and document storage for scalable applications  
✔ **Features**:  
   - Supports **ACID transactions**  
   - **NoSQL + SQL hybrid** database  
   - Supports **native Java API & SQL-like queries**  
✔ **Example Use Case**: Used in **big data** and **graph-based applications**  
✔ **Website**: [orientdb.org](https://orientdb.org/)  

---

### **5️⃣ Neo4j**  
✔ **Type**: Graph Database (NoSQL)  
✔ **Use Case**: Social networks, recommendation engines, fraud detection  
✔ **Features**:  
   - Optimized for **graph-based queries**  
   - Uses **Cypher Query Language (CQL)**  
   - Provides **native Java APIs**  
✔ **Example Use Case**: **Social network connections, fraud detection**  
✔ **Website**: [neo4j.com](https://neo4j.com/)  

---

### **6️⃣ Apache Cassandra** *(Partially in Java, Core in Java)*  
✔ **Type**: NoSQL (Column-Oriented, Distributed)  
✔ **Use Case**: Large-scale distributed storage (Big Data)  
✔ **Features**:  
   - High availability and fault tolerance  
   - Scales horizontally across multiple nodes  
   - Supports **Java-based client APIs**  
✔ **Example Use Case**: Used by **Netflix, Facebook, Twitter** for **high-traffic applications**  
✔ **Website**: [cassandra.apache.org](https://cassandra.apache.org/)  

---

### **7️⃣ Elasticsearch** *(Partially in Java, Core in Java)*  
✔ **Type**: Search Engine + NoSQL Document Store  
✔ **Use Case**: Full-text search, log analytics, real-time indexing  
✔ **Features**:  
   - **Distributed, scalable, and fast**  
   - Supports **REST API & JSON queries**  
   - **Built-in clustering & sharding**  
✔ **Example Use Case**: **Log analytics (ELK Stack), e-commerce search, AI applications**  
✔ **Website**: [elastic.co](https://www.elastic.co/)  

---

## **🚀 Summary Table**
| Database | Type | Best For | Use Case |
|----------|------|---------|----------|
| **H2 Database** | Relational (SQL) | In-memory, lightweight apps | Spring Boot, testing |
| **Apache Derby** | Relational (SQL) | Embedded Java DB | Java SE (JavaDB) |
| **HSQLDB** | Relational (SQL) | In-memory & fast processing | JUnit testing |
| **OrientDB** | Multi-model (Graph, Document) | NoSQL + SQL hybrid | Big data, complex relationships |
| **Neo4j** | Graph Database | Social networks, recommendations | Fraud detection, AI |
| **Apache Cassandra** | NoSQL (Columnar) | Distributed, high availability | Big data (Netflix, Twitter) |
| **Elasticsearch** | NoSQL (Search, Document) | Full-text search & logs | Log analytics (ELK Stack) |

---

## **🎯 Which Java Database Should You Use?**
- **For testing & lightweight apps →** H2, HSQLDB, Apache Derby  
- **For relational storage in Java apps →** H2, Apache Derby  
- **For graph-based data (social networks) →** Neo4j, OrientDB  
- **For distributed, high-scale NoSQL storage →** Apache Cassandra  
- **For search engines & analytics →** Elasticsearch  

Would you like a sample **Java integration example** for any of these databases?