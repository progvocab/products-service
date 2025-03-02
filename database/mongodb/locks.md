### **Do NoSQL Databases Like MongoDB Use Locks Like PostgreSQL?**  

Yes, **NoSQL databases like MongoDB also use locks**, but they handle them differently from PostgreSQL. Unlike PostgreSQL, which relies on **heavyweight locks (row-level, table-level, transaction locks)** for strict ACID compliance, MongoDB uses a **more lightweight and flexible locking mechanism** optimized for high concurrency.

---

## **1. Locking in MongoDB vs. PostgreSQL**  

| Feature               | **MongoDB** (NoSQL) | **PostgreSQL** (SQL) |
|----------------------|-------------------|--------------------|
| **Locking Granularity** | **Document-level locks** | **Row-level, table-level, transaction locks** |
| **Locking Type** | **Reader-Writer locks** | **Heavyweight transactional locks** |
| **Concurrency Control** | **Optimistic & Pessimistic** | **MVCC (Multi-Version Concurrency Control)** |
| **Transactions** | Supported (but limited compared to SQL) | Full ACID-compliant transactions |
| **Default Behavior** | Uses **locks only when necessary** | Uses **locks for every transaction** |

---

## **2. Locking Mechanisms in MongoDB**  

### **(a) Document-Level Locking**  
- MongoDB **locks only the document** being modified instead of an entire table or row.  
- This allows **multiple clients to update different documents** in the same collection without blocking.  

#### **Example: Updating Two Documents Simultaneously**  
```javascript
db.users.updateOne({ _id: 1 }, { $set: { name: "Alice" } });
db.users.updateOne({ _id: 2 }, { $set: { name: "Bob" } });
```
- Each update acquires a **separate lock for each document**, allowing high concurrency.  

---

### **(b) Collection-Level Locking** (Rare Cases)  
- In **some operations (e.g., index builds, schema changes)**, MongoDB locks the **entire collection** to maintain consistency.  
- This is **less granular** than PostgreSQL's row-level locking but prevents corruption.  

#### **Example: Creating an Index (Causes Collection Lock)**
```javascript
db.users.createIndex({ email: 1 });
```
- This temporarily **locks the entire `users` collection** until the index is built.

---

### **(c) Global Locking (Extremely Rare)**  
- If a **major operation like journaling, replication sync, or shutdown** occurs, MongoDB **locks the entire database instance**.
- This is **very rare** and usually happens only in administrative tasks.

#### **Example: Database Shutdown (Triggers Global Lock)**
```javascript
db.adminCommand({ shutdown: 1 });
```
- This **locks the entire MongoDB instance** until shutdown is complete.

---

## **3. Locking in PostgreSQL (Compared to MongoDB)**  
PostgreSQL uses **Multi-Version Concurrency Control (MVCC)** instead of direct document locks.  
- **PostgreSQL maintains multiple versions of a row**, allowing reads and writes to happen without blocking each other.  
- However, **PostgreSQL requires locks for transactions**, whereas MongoDB avoids unnecessary locking.  

#### **Example: PostgreSQL Row-Level Lock (Similar to MongoDB Document Locking)**
```sql
BEGIN;
UPDATE users SET name = 'Alice' WHERE id = 1;
COMMIT;
```
- Locks **only row ID 1** (similar to MongoDB's document lock).  

---

## **4. How MongoDB Avoids Locking Issues**
### ✅ **Uses WiredTiger Storage Engine (Optimized for Concurrent Writes)**  
- Supports **document-level locking** instead of table-wide locks.  
- Uses **snapshot isolation (similar to MVCC in PostgreSQL)** for high concurrency.  

### ✅ **Uses Journal-Based Writes Instead of Strict WAL**
- MongoDB uses **journaling** to ensure durability, reducing the need for transaction-level locking.  
- PostgreSQL uses **WAL (Write-Ahead Logging)**, which requires **frequent synchronization locks**.

---

## **5. When Does MongoDB Use Transactions Like PostgreSQL?**  
- Since **MongoDB 4.0+, multi-document transactions** are supported, similar to SQL databases.  
- When using **multi-document transactions**, MongoDB **locks multiple documents at once**, similar to row locks in PostgreSQL.

#### **Example: MongoDB Multi-Document Transaction (Uses Locks)**
```javascript
const session = db.getMongo().startSession();
session.startTransaction();
db.users.updateOne({ _id: 1 }, { $set: { balance: 500 } }, { session });
db.accounts.updateOne({ _id: 101 }, { $inc: { funds: -500 } }, { session });
session.commitTransaction();
```
- This **locks both `users` and `accounts` documents** until the transaction commits.  
- Similar to a **PostgreSQL transaction lock**.  

---

## **6. Conclusion: Do NoSQL Databases Like MongoDB Use Locks Like PostgreSQL?**
✅ **Yes, MongoDB uses locks**, but they are:  
- **More lightweight** (document-level instead of row/table-level).  
- **Less frequent** (locks only when necessary).  
- **Optimized for high-concurrency workloads** (WiredTiger engine, reader-writer locks).  
- **PostgreSQL requires strict transactional locks**, while MongoDB prefers **optimistic concurrency**.  

Would you like **performance tuning tips for reducing MongoDB lock contention?**