### **Types of Indexes in PostgreSQL (With Examples)**  

PostgreSQL supports several types of indexes, each optimized for different use cases. Choosing the right index can **improve query performance** significantly.

---

## **1. B-Tree Index (Default)**
- **Best for:** Equality (`=`), Range (`<`, `>`, `BETWEEN`), Sorting (`ORDER BY`).
- **Balanced tree structure** ensures **logarithmic time complexity** (`O(log n)`).
- Automatically created for **PRIMARY KEY** and **UNIQUE constraints**.

### **Example: Creating a B-Tree Index**
```sql
CREATE INDEX idx_employee_salary ON employees (salary);
```
### **Query Optimization**
```sql
EXPLAIN ANALYZE
SELECT * FROM employees WHERE salary > 50000;
```
- The query will use `idx_employee_salary` for fast lookup.

---

## **2. Hash Index**
- **Best for:** **Exact match (`=`) queries**.
- **Not useful for range queries**.
- Works like a **hash table** (constant-time lookup).

### **Example: Creating a Hash Index**
```sql
CREATE INDEX idx_employee_id ON employees USING HASH(emp_id);
```
### **Query Optimization**
```sql
EXPLAIN ANALYZE
SELECT * FROM employees WHERE emp_id = 1001;
```
- Uses `idx_employee_id` for direct lookup.

**⚠️ Note:** Hash indexes were unreliable before PostgreSQL 10, but now they support WAL logging and replication.

---

## **3. GIN (Generalized Inverted Index)**
- **Best for:** **Full-text search, JSONB, Arrays**.
- Stores **keys mapped to row locations** for **fast lookups**.

### **Example: Full-Text Search**
```sql
CREATE INDEX idx_search ON documents USING GIN(to_tsvector('english', content));
```
### **Query Optimization**
```sql
SELECT * FROM documents WHERE to_tsvector('english', content) @@ to_tsquery('database');
```
- **Faster search** in large text datasets.

---

## **4. GiST (Generalized Search Tree)**
- **Best for:** **Geospatial, Range Queries, Fuzzy Search**.
- Supports **R-tree-like** indexing.

### **Example: Geospatial Indexing (PostGIS)**
```sql
CREATE INDEX idx_geo ON locations USING GiST(geom);
```
### **Query Optimization**
```sql
SELECT * FROM locations WHERE geom && ST_MakeEnvelope(-75, 40, -73, 42, 4326);
```
- **Finds points inside a geographic bounding box**.

---

## **5. SP-GiST (Space-Partitioned GiST)**
- **Best for:** **High-dimensional data (QuadTrees, KD-Trees, Text search)**.
- Used for **text indexing, IP ranges, and hierarchical data**.

### **Example: IP Address Indexing**
```sql
CREATE INDEX idx_ip ON users USING SPGIST(ip_address);
```
### **Query Optimization**
```sql
SELECT * FROM users WHERE ip_address <<= '192.168.1.0/24';
```
- **Efficiently finds IPs within a subnet**.

---

## **6. BRIN (Block Range INdex)**
- **Best for:** **Very large tables with sorted columns** (e.g., time-series).
- Stores **metadata about row ranges** instead of individual values.

### **Example: BRIN for Time-Series Data**
```sql
CREATE INDEX idx_logs_date ON logs USING BRIN(log_date);
```
### **Query Optimization**
```sql
SELECT * FROM logs WHERE log_date BETWEEN '2024-01-01' AND '2024-02-01';
```
- **Faster than a B-Tree for large datasets**.

---

## **7. Covering Index (INCLUDE Clause)**
- **Best for:** **Optimizing queries without fetching extra data**.
- **Stores additional columns** in the index to avoid accessing the table.

### **Example: Covering Index**
```sql
CREATE INDEX idx_orders ON orders(customer_id) INCLUDE(order_date, total_price);
```
### **Query Optimization**
```sql
SELECT customer_id, order_date, total_price FROM orders WHERE customer_id = 101;
```
- **Prevents unnecessary table access**, improving performance.

---

## **8. Partial Index**
- **Best for:** **Indexing only a subset of data**.
- Reduces index size and improves performance.

### **Example: Partial Index on Active Users**
```sql
CREATE INDEX idx_active_users ON users(email) WHERE is_active = TRUE;
```
### **Query Optimization**
```sql
SELECT * FROM users WHERE email = 'test@example.com' AND is_active = TRUE;
```
- **Only indexes active users**, making queries faster.

---

## **9. Expression Index**
- **Best for:** **Indexing computed expressions**.

### **Example: Index on Lowercase Email**
```sql
CREATE INDEX idx_lower_email ON users (LOWER(email));
```
### **Query Optimization**
```sql
SELECT * FROM users WHERE LOWER(email) = 'test@example.com';
```
- **Faster case-insensitive email lookups**.

---

## **10. Unique Index**
- **Ensures uniqueness** of values in a column.

### **Example: Unique Index on Username**
```sql
CREATE UNIQUE INDEX idx_unique_username ON users(username);
```
### **Query Optimization**
```sql
INSERT INTO users (username) VALUES ('john_doe');
```
- **Prevents duplicate usernames**.

---

## **Summary: When to Use Each Index?**
| Index Type | **Best For** |
|------------|------------|
| **B-Tree** (Default) | General-purpose queries (`=`, `<`, `>`, `BETWEEN`) |
| **Hash** | Exact matches (`=`) |
| **GIN** | JSONB, Arrays, Full-text search |
| **GiST** | Geospatial, Fuzzy Search, Range Queries |
| **SP-GiST** | Hierarchical, IP Address, KD-Trees |
| **BRIN** | Large tables with sorted data (e.g., time-series) |
| **Covering (INCLUDE)** | Index-only scans, reducing table access |
| **Partial** | Indexing only a subset of rows |
| **Expression** | Indexing computed values (e.g., `LOWER(email)`) |
| **Unique** | Enforcing uniqueness |

Would you like recommendations on indexing for your PostgreSQL application?