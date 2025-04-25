Oracle supports a wide range of **index types**, each designed to optimize different kinds of queries and data patterns. Here’s a breakdown of the **most commonly used index types in Oracle**, along with their **use cases**:

---

### **1. B-tree Index (Default)**
- **Structure**: Balanced tree structure
- **Use Case**: Fast lookup on columns with high cardinality (many distinct values)
- **Example**:
  ```sql
  CREATE INDEX idx_employee_name ON employee(name);
  ```

---

### **2. Bitmap Index**
- **Structure**: Uses bitmap vectors for each key value
- **Use Case**: Ideal for columns with **low cardinality** (few distinct values) like gender, status flags
- **Example**:
  ```sql
  CREATE BITMAP INDEX idx_gender ON employee(gender);
  ```
- **Note**: Not suitable for high-concurrency environments (DML-heavy workloads)

---

### **3. Unique Index**
- **Use Case**: Ensures uniqueness on one or more columns
- **Example**:
  ```sql
  CREATE UNIQUE INDEX idx_emp_id ON employee(emp_id);
  ```

---

### **4. Composite Index (Concatenated Index)**
- **Use Case**: Index on multiple columns (order matters)
- **Example**:
  ```sql
  CREATE INDEX idx_emp_name_dept ON employee(last_name, department_id);
  ```

---

### **5. Function-Based Index**
- **Use Case**: Index on expressions or functions to speed up computed WHERE clauses
- **Example**:
  ```sql
  CREATE INDEX idx_upper_name ON employee(UPPER(name));
  ```

---

### **6. Reverse Key Index**
- **Use Case**: Reverses bytes of the key to avoid index hot spots (useful for sequential inserts)
- **Example**:
  ```sql
  CREATE INDEX idx_rev_emp_id ON employee(emp_id) REVERSE;
  ```

---

### **7. Domain Index**
- **Use Case**: Custom index for complex data types like spatial, text, XML
- **Example**:
  Oracle Text, Oracle Spatial use this under the hood.

---

### **8. Clustered Index (IOT - Index Organized Table)**
- **Use Case**: Table data stored in index structure itself; no separate table heap
- **Example**:
  ```sql
  CREATE TABLE employee (
    emp_id NUMBER PRIMARY KEY,
    name VARCHAR2(50)
  ) ORGANIZATION INDEX;
  ```

---

### **9. Global vs Local Partitioned Indexes**
- **Local Index**: One index per partition (partitioned table)
- **Global Index**: One index across all partitions

---

### Summary Table

| Index Type         | Best For                         | Notes |
|--------------------|----------------------------------|-------|
| B-tree             | High-cardinality queries         | Default and general-purpose |
| Bitmap             | Low-cardinality, analytical use  | Not good for OLTP |
| Unique             | Enforcing uniqueness             | Same as B-tree with unique constraint |
| Composite          | Multi-column lookups             | Column order matters |
| Function-based     | Queries using expressions        | Must use deterministic functions |
| Reverse Key        | Avoiding hot index blocks        | Useful in high-insert environments |
| Domain             | Custom data types (text/spatial) | Needs special libraries |
| IOT (Clustered)    | Faster primary key access        | Entire row is in the index |
| Partitioned Index  | Partitioned tables               | Local and global variants |

---

Let me know if you’d like to see performance examples or how to monitor index usage!