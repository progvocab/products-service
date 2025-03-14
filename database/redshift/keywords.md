### **Redshift-Specific SQL Keywords & Features**  

Amazon Redshift is based on **PostgreSQL**, but it has some **unique SQL keywords and commands** that are optimized for **distributed, columnar storage and MPP (Massively Parallel Processing) architecture**.  

---

## **🔹 Redshift-Only Keywords & Features**  
### **1️⃣ DISTRIBUTION KEYS & SORT KEYS**
- **`DISTKEY`** – Defines how data is distributed across nodes.  
- **`SORTKEY`** – Defines how data is stored for optimized queries.  
- **`INTERLEAVED SORTKEY`** – Optimizes multiple sort keys.  

✅ **Example: Creating a table with DISTKEY and SORTKEY**
```sql
CREATE TABLE sales (
    sale_id INT,
    customer_id INT DISTKEY,
    sale_date DATE SORTKEY
);
```
---

### **2️⃣ UNLOAD & COPY (For Fast Data Transfer)**
- **`UNLOAD`** – Exports data from Redshift to S3.  
- **`COPY`** – Imports data from S3, DynamoDB, or other sources.  

✅ **Example: Load data from S3 using COPY**
```sql
COPY sales FROM 's3://my-bucket/sales-data/'
CREDENTIALS 'aws_iam_role=arn:aws:iam::123456789012:role/MyRedshiftRole'
FORMAT AS PARQUET;
```
✅ **Example: Export data to S3 using UNLOAD**
```sql
UNLOAD ('SELECT * FROM sales')
TO 's3://my-bucket/exported-sales/'
IAM_ROLE 'arn:aws:iam::123456789012:role/MyRedshiftRole'
FORMAT AS PARQUET;
```
---

### **3️⃣ ENCODE (Columnar Compression)**
- **`ENCODE`** – Defines compression encoding for each column.  
- Redshift **automatically applies compression** based on the column data type.  

✅ **Example: Creating a table with column encoding**
```sql
CREATE TABLE users (
    user_id INT ENCODE ZSTD,
    username VARCHAR(100) ENCODE LZO,
    signup_date DATE ENCODE DELTA
);
```
---

### **4️⃣ MATERIALIZED VIEWS (Optimized Query Performance)**
- **`MATERIALIZED VIEW`** – Stores precomputed query results to improve performance.  

✅ **Example: Creating a Materialized View**
```sql
CREATE MATERIALIZED VIEW mv_sales AS
SELECT customer_id, SUM(amount) AS total_spent
FROM sales
GROUP BY customer_id;
```
✅ **Example: Refreshing a Materialized View**
```sql
REFRESH MATERIALIZED VIEW mv_sales;
```
---

### **5️⃣ WORKLOAD MANAGEMENT (WLM)**
- **`WLM_QUEUE_ASSIGNMENT`** – Assigns queries to different workload queues for performance tuning.  

✅ **Example: Assigning a session to a workload queue**
```sql
SET wlm_queue_assignment = 'high_priority';
```
---

### **6️⃣ RA3 & AQUA-SPECIFIC COMMANDS**
- **`AQUA ENABLED`** – Enables **Amazon Redshift AQUA**, a hardware-accelerated query optimization feature.  
- **`DATASHARING`** – Used for **cross-account data sharing in Redshift**.  

✅ **Example: Enable data sharing**
```sql
ALTER DATASHARE my_share ADD SCHEMA public;
```
---

### **7️⃣ CROSS-DATABASE QUERIES (Only in Redshift)**
- **`CREATE DATABASE LINK`** – Enables querying across multiple Redshift databases.  

✅ **Example: Querying another database in Redshift**
```sql
SELECT * FROM my_other_database.public.sales;
```
---

### **🔹 Summary: Redshift-Only Keywords**
| **Feature** | **Keywords** |
|------------|-------------|
| **Data Distribution** | `DISTKEY`, `SORTKEY`, `INTERLEAVED SORTKEY` |
| **Data Ingestion & Export** | `COPY`, `UNLOAD` |
| **Compression & Storage** | `ENCODE` |
| **Performance Optimization** | `MATERIALIZED VIEW`, `WLM_QUEUE_ASSIGNMENT` |
| **Advanced Features** | `AQUA ENABLED`, `DATASHARING`, `CREATE DATABASE LINK` |

Would you like a **performance tuning guide** for Redshift queries? 🚀