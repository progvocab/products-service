# Locks




Locks are used to **control concurrent access** to database objects so that:

* Data consistency is maintained
* Conflicting operations do not corrupt data

PostgreSQL uses a **MVCC (Multi-Version Concurrency Control)** model, so:

* Readers do **not block writers**
* Writers do **not block readers**
* Conflicts mainly occur **between writers**


### Lock Scope

* **Row-level locks**
* **Table-level locks**
* **Page-level locks (internal)**
* **Transaction-level locks**
* **Advisory locks (application-controlled)**


### Row-Level Locks 
Most Common ,Row locks are acquired during `INSERT`, `UPDATE`, `DELETE`, or `SELECT … FOR UPDATE`.

| Lock Type         | Description                     | Conflicts With      |
| ----------------- | ------------------------------- | ------------------- |
| FOR UPDATE        | Exclusive lock on selected rows | All other row locks |
| FOR NO KEY UPDATE | Prevents key changes            | FOR UPDATE          |
| FOR SHARE         | Read lock                       | FOR UPDATE          |
| FOR KEY SHARE     | Protects referenced keys        | FOR UPDATE          |



```sql
BEGIN;
SELECT * FROM orders WHERE id = 10 FOR UPDATE;
```

* Blocks other updates/deletes on the same row , avoid race condition
* Does **not** block plain `SELECT`
```sql
SELECT * FROM orders WHERE id = 10;
```

but if there is another request , using `UPDATE` or `FOR UPDATE`
```
UPDATE orders SET status = 'SHIPPED' WHERE id = 10;
```
this is BLOCKED
- The UPDATE tries to acquire a row lock
- PostgreSQL puts the request into a wait queue No error is thrown Waits until:
  + First transaction commits → UPDATE proceeds
  + First transaction rolls back → UPDATE proceeds
  + Lock timeout occurs → ERROR
```sql
SET lock_timeout = '3s';
```
Use the above statement to configure lock timeout


```sql
SELECT * FROM orders WHERE id = 10 FOR UPDATE NOWAIT;
```
Using `NOWAIT` throws error , the update does not wait or block , it is either success or failure

```sh
ERROR: could not obtain lock on row
```


###  Table-Level Locks

Automatically acquired depending on the SQL operation.

| Lock Mode              | Acquired By                       | Blocks         |
| ---------------------- | --------------------------------- | -------------- |
| ACCESS SHARE           | SELECT                            | DROP, TRUNCATE |
| ROW SHARE              | SELECT FOR UPDATE                 | EXCLUSIVE      |
| ROW EXCLUSIVE          | INSERT, UPDATE, DELETE            | VACUUM FULL    |
| SHARE UPDATE EXCLUSIVE | VACUUM, CREATE INDEX CONCURRENTLY | DDL            |
| EXCLUSIVE              | DELETE all rows                   | ACCESS SHARE   |
| ACCESS EXCLUSIVE       | ALTER, DROP, TRUNCATE             | Everything     |



```sql
ALTER TABLE employees ADD COLUMN age INT;
```

* Acquires **ACCESS EXCLUSIVE**
* Blocks **all reads and writes**


### Transaction-Level Locks

Used internally to ensure transaction consistency.

| Lock                | Purpose                      |
| ------------------- | ---------------------------- |
| Transaction ID lock | Ensures visibility rules     |
| Virtual XID lock    | Prevents conflicting commits |

You normally **do not manage these explicitly**.


### Advisory Locks (Application-Controlled)

Explicit locks managed by the application.

| Function                  | Scope             |
| ------------------------- | ----------------- |
| pg_advisory_lock(id)      | Session-level     |
| pg_advisory_xact_lock(id) | Transaction-level |



```sql
SELECT pg_advisory_xact_lock(1001);
```

Use cases:

* Distributed schedulers
* Leader election
* Preventing duplicate batch jobs (very useful with Spring Batch)


## 7. How Deadlocks Happen

Deadlocks occur when transactions **wait on each other cyclically**.

Example:

```text
T1 locks row A → wants row B
T2 locks row B → wants row A
```

PostgreSQL:

* Detects deadlock automatically
* Aborts **one transaction**
* Throws error:

```text
ERROR: deadlock detected
```


### Diagnosing Locks and Blocking

### Find Blocking Queries

```sql
SELECT
  pid,
  usename,
  state,
  wait_event_type,
  wait_event,
  query
FROM pg_stat_activity
WHERE wait_event IS NOT NULL;
```

### Who Is Blocking Whom

```sql
SELECT
  blocked.pid     AS blocked_pid,
  blocking.pid    AS blocking_pid,
  blocked.query   AS blocked_query,
  blocking.query  AS blocking_query
FROM pg_stat_activity blocked
JOIN pg_stat_activity blocking
ON blocking.pid = ANY(pg_blocking_pids(blocked.pid));
```


### Lock Timeout and Safety Controls

### Set Lock Timeout

```sql
SET lock_timeout = '5s';
```

Prevents sessions from waiting indefinitely.



Statement Timeout

```sql
SET statement_timeout = '30s';
```

Kills long-running queries.


### Best Practices to Avoid Lock Issues

* Keep transactions **short**
* Avoid `SELECT FOR UPDATE` unless required
* Always access tables in **same order** across services
* Avoid `VACUUM FULL` in production
* Use `CREATE INDEX CONCURRENTLY`
* Prefer **advisory locks** for business coordination
* Monitor `pg_stat_activity` continuously

 

### PostgreSQL Locks vs Other Databases

| Feature                 | PostgreSQL | MySQL (InnoDB) | Oracle    |
| ----------------------- | ---------- | -------------- | --------- |
| MVCC                    | Yes        | Partial        | Yes       |
| Readers block writers   | No         | Sometimes      | No        |
| Explicit advisory locks | Yes        | Limited        | No        |
| Deadlock detection      | Automatic  | Automatic      | Automatic |
 
### Common Real-World Scenarios

* Spring Batch job stuck → **advisory lock not released**
* ALTER TABLE hangs → **ACCESS EXCLUSIVE lock**
* Kafka consumer lag → **long-running transaction**
* API latency spikes → **row lock contention**

More :

* PostgreSQL locks **specifically for Spring Batch**
* How locks behave in **REPEATABLE READ vs SERIALIZABLE**
* Locking impact of **indexes and foreign keys**
* How to simulate and reproduce deadlocks
 


### **PostgreSQL Locks: `lock:transactionid`, `lock:tuple`, and Others**  
PostgreSQL uses **locks** to ensure **data consistency, prevent race conditions, and manage concurrency**. Different types of locks help manage transactions, rows, tables, and other database objects.  



### **1. `lock:transactionid` (Transaction ID Lock)**

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



### **2. `lock:tuple` (Row-Level Lock)**

- Acquired **when a transaction updates or deletes a row**.  
- Ensures that no other transaction modifies the same row until the first transaction **commits or rolls back**.  


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



### **Other Important Locks in PostgreSQL**
| **Lock Type**          | **Description** |
|------------------------|----------------|
| **`lock:relation`**    | Acquired when a transaction accesses a **table** (e.g., `SELECT`, `INSERT`, `UPDATE`). |
| **`lock:page`**        | Rarely used; locks a **specific page in a table or index**. |
| **`lock:extend`**      | Used when a transaction needs to **extend a relation (table growth)**. |
| **`lock:virtualxid`**  | Ensures a transaction **remains active until a specific operation is complete**. |
| **`lock:advisory`**    | User-defined locks for **custom locking scenarios** (e.g., `pg_advisory_lock()`). |



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
