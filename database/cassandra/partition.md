In **Apache Cassandra**, **per-partition limit** is a query feature that allows you to limit the number of rows returned **per partition** in a query. This is particularly useful when dealing with **wide partitions** (partitions with many rows) and you want to avoid fetching excessive data. Let’s explore this concept in detail, along with related concepts like **paging**, **token function**, and **partition keys**.

---

### **What is a Partition in Cassandra?**
- A **partition** in Cassandra is a group of rows that share the same **partition key**.
- Data within a partition is sorted by the **clustering columns**.
- Partitions are distributed across nodes in the cluster based on the **token ring**.

---

### **Per-Partition Limit**
- The **per-partition limit** restricts the number of rows returned **per partition** in a query.
- This is different from the **global limit**, which limits the total number of rows returned by the query.

#### **Syntax**:
```sql
SELECT * FROM table_name
WHERE partition_key = ?
PER PARTITION LIMIT N;
```
- `N` is the maximum number of rows to return per partition.

#### **Example**:
Suppose you have a table `user_activity` with the following schema:
```sql
CREATE TABLE user_activity (
    user_id UUID,
    activity_time TIMESTAMP,
    activity_type TEXT,
    details TEXT,
    PRIMARY KEY (user_id, activity_time)
);
```

To fetch the **latest 5 activities** for each user:
```sql
SELECT * FROM user_activity
PER PARTITION LIMIT 5;
```

---

### **Why Use Per-Partition Limit?**
1. **Efficient Querying**:
   - Prevents fetching excessive rows from wide partitions, improving query performance.

2. **Pagination**:
   - Helps implement pagination by limiting the number of rows per partition.

3. **Resource Management**:
   - Reduces memory and network overhead by limiting the amount of data returned.

---

### **Related Concepts**

#### **1. Paging**
- **Paging** allows you to fetch large result sets in smaller chunks (pages) to avoid overwhelming the client or server.
- Cassandra automatically enables paging for queries that return more than a certain number of rows (default: 5000).

#### **Example**:
```sql
SELECT * FROM user_activity
WHERE user_id = ?
PER PARTITION LIMIT 5;
```
- If the result set is large, Cassandra will return the first page of results and provide a **paging state** to fetch the next page.

#### **Using Paging in Drivers**:
Most Cassandra drivers (e.g., Java, Python) support paging. For example, in Python:
```python
from cassandra.cluster import Cluster

cluster = Cluster()
session = cluster.connect('my_keyspace')

query = "SELECT * FROM user_activity PER PARTITION LIMIT 5"
rows = session.execute(query, paging_state=None)

for row in rows:
    print(row)

# Fetch the next page
paging_state = rows.paging_state
rows = session.execute(query, paging_state=paging_state)
```

---

#### **2. Token Function**
- The **token function** is used to query data based on the **token value** of the partition key.
- This is useful for low-level operations like **data distribution analysis** or **custom partitioning**.

#### **Example**:
```sql
SELECT * FROM user_activity
WHERE token(user_id) = token(?);
```
- This query fetches rows for a specific token value.

---

#### **3. Partition Key**
- The **partition key** determines how data is distributed across nodes in the cluster.
- It is the first part of the primary key and is used to identify the partition.

#### **Example**:
In the `user_activity` table:
```sql
PRIMARY KEY (user_id, activity_time)
```
- `user_id` is the partition key.
- `activity_time` is the clustering column.

---

### **Example Use Cases**

#### **1. Fetching Recent Activity**
To fetch the **latest 3 activities** for each user:
```sql
SELECT * FROM user_activity
PER PARTITION LIMIT 3;
```

#### **2. Pagination with Per-Partition Limit**
To fetch the **next 3 activities** for each user using paging:
```sql
SELECT * FROM user_activity
PER PARTITION LIMIT 3;
```
- Use the paging state to fetch the next page of results.

#### **3. Filtering by Token**
To fetch rows for a specific token value:
```sql
SELECT * FROM user_activity
WHERE token(user_id) = token(?);
```

---

### **Best Practices**
1. **Avoid Wide Partitions**:
   - Design your data model to avoid wide partitions, as they can lead to performance issues.

2. **Use Per-Partition Limit for Wide Partitions**:
   - Use `PER PARTITION LIMIT` to limit the number of rows fetched from wide partitions.

3. **Combine with Paging**:
   - Use paging to handle large result sets efficiently.

4. **Monitor Query Performance**:
   - Use tools like `nodetool` to monitor query performance and identify wide partitions.

---

### **Summary**
- **Per-partition limit** restricts the number of rows returned per partition in a query.
- It is useful for querying wide partitions, implementing pagination, and managing resources.
- Related concepts include **paging**, **token function**, and **partition keys**.
- Use `PER PARTITION LIMIT` to optimize queries and improve performance in Cassandra.

Let me know if you need further clarification!