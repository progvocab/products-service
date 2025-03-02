### **PostgreSQL Locks: `lock:transactionid`, `lock:tuple`, and Others**  
PostgreSQL uses **locks** to ensure **data consistency, prevent race conditions, and manage concurrency**. Different types of locks help manage transactions, rows, tables, and other database objects.  

---

## **1. `lock:transactionid` (Transaction ID Lock)**
### **What It Does:**  
- Acquired **when one transaction needs to wait for another transaction to complete**.  
- Happens when a transaction is trying to modify or read data that another transaction is **still using**.  

### **Example Scenario:**  
```sql
-- Transaction 1 (T1): Starts and updates a row without committing
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;

-- Transaction 2 (T2): Tries to read the same row but gets blocked
SELECT * FROM accounts WHERE id = 1;
```
- **T1 holds a `lock:transactionid`**, preventing other transactions from reading/writing.  
- **T2 is blocked until T1 commits or rolls back**.  

### **How to Identify `lock:transactionid` Issues?**
```sql
SELECT * FROM pg_locks WHERE locktype = 'transactionid';
```
- If a transaction **holds this lock for too long**, it may cause blocking and slow performance.

---

## **2. `lock:tuple` (Row-Level Lock)**
### **What It Does:**  
- Acquired **when a transaction updates or deletes a row**.  
- Ensures that no other transaction modifies the same row until the first transaction **commits or rolls back**.  

### **Example Scenario:**
```sql
-- Transaction 1 (T1): Locks a row for update
BEGIN;
UPDATE employees SET salary = salary + 500 WHERE id = 10;

-- Transaction 2 (T2): Tries to update the same row (BLOCKED)
UPDATE employees SET salary = salary + 1000 WHERE id = 10;
```
- **T1 holds a `lock:tuple` on row ID 10**.  
- **T2 is blocked until T1 commits or rolls back**.  

### **How to Detect `lock:tuple` Issues?**
```sql
SELECT * FROM pg_locks WHERE locktype = 'tuple';
```
- High tuple locks may indicate **long-running updates** that are blocking queries.  

---

## **Other Important Locks in PostgreSQL**
| **Lock Type**          | **Description** |
|------------------------|----------------|
| **`lock:relation`**    | Acquired when a transaction accesses a **table** (e.g., `SELECT`, `INSERT`, `UPDATE`). |
| **`lock:page`**        | Rarely used; locks a **specific page in a table or index**. |
| **`lock:extend`**      | Used when a transaction needs to **extend a relation (table growth)**. |
| **`lock:virtualxid`**  | Ensures a transaction **remains active until a specific operation is complete**. |
| **`lock:advisory`**    | User-defined locks for **custom locking scenarios** (e.g., `pg_advisory_lock()`). |

---

## **How to Monitor Locks in PostgreSQL?**
### **1. Check Active Locks**
```sql
SELECT pid, relation::regclass, mode, granted
FROM pg_locks
WHERE NOT granted;
```
- Shows **queries waiting for locks**.

### **2. Detect Blocking Transactions**
```sql
SELECT blocking.pid AS blocking_pid, blocked.pid AS blocked_pid,
       blocking.query AS blocking_query, blocked.query AS blocked_query
FROM pg_stat_activity blocked
JOIN pg_locks blocked_locks ON blocked.pid = blocked_locks.pid
JOIN pg_locks blocking_locks ON blocked_locks.locktype = blocking_locks.locktype
AND blocked_locks.database IS NOT DISTINCT FROM blocking_locks.database
AND blocked_locks.relation IS NOT DISTINCT FROM blocking_locks.relation
AND blocked_locks.page IS NOT DISTINCT FROM blocking_locks.page
AND blocked_locks.tuple IS NOT DISTINCT FROM blocking_locks.tuple
AND blocked_locks.transactionid IS NOT DISTINCT FROM blocking_locks.transactionid
AND blocked_locks.classid IS NOT DISTINCT FROM blocking_locks.classid
AND blocked_locks.objid IS NOT DISTINCT FROM blocking_locks.objid
AND blocked_locks.objsubid IS NOT DISTINCT FROM blocking_locks.objsubid
JOIN pg_stat_activity blocking ON blocking_locks.pid = blocking.pid
WHERE NOT blocked_locks.granted;
```
- Identifies **which query is blocking others**.

---

### **How to Reduce Locking Issues?**
1. **Optimize Transactions:**
   - Keep **transactions short** to release locks quickly.
   - Avoid unnecessary `BEGIN;` statements without `COMMIT;`.

2. **Use `FOR UPDATE SKIP LOCKED` (Non-blocking Reads):**
   ```sql
   SELECT * FROM orders WHERE status = 'pending' FOR UPDATE SKIP LOCKED;
   ```
   - Allows other transactions to process unlocked rows.

3. **Tune Autovacuum & Freezing:**
   ```sql
   VACUUM ANALYZE;
   ```
   - Prevents **wraparound issues** that can cause high transaction ID locks.

---

### **Conclusion**
- **`lock:transactionid`** → Locks a **transaction**, preventing conflicts.  
- **`lock:tuple`** → Locks a **specific row** for updates.  
- PostgreSQL has **multiple lock types** for managing concurrent access to tables, pages, and rows.  
- **Monitoring locks** and **optimizing queries** can help reduce contention.  

Would you like an example of **tuning PostgreSQL locks for high-concurrency workloads**?