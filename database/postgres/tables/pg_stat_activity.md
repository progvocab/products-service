# pg_stat_activity  

`pg_stat_activity` is **the most important PostgreSQL system view for observing what is happening *right now*** in the database.

Think of it as **PostgreSQL’s live process monitor** (similar to `top` for Linux).
 

## 1. What Is `pg_stat_activity`

`pg_stat_activity` is a **system catalog view** that shows:

* Every active backend process
* Who is connected
* What each session is doing
* What queries are running
* Who is waiting and why

```sql
SELECT * FROM pg_stat_activity;
```

Each row = **one database session / connection**

---

## 2. Why It Exists (Primary Use Cases)

You use `pg_stat_activity` to:

* Debug **blocking and locks**
* Find **slow or stuck queries**
* Identify **idle-in-transaction** sessions
* Monitor **connection usage**
* Diagnose **deadlocks and timeouts**
* Kill problematic sessions safely

---

## 3. Key Columns You Must Know

### Session Identity

| Column           | Meaning               |
| ---------------- | --------------------- |
| pid              | Process ID of backend |
| usename          | Database user         |
| datname          | Database name         |
| application_name | App / service name    |
| client_addr      | Client IP             |

---

### Query State

| Column        | Meaning                             |
| ------------- | ----------------------------------- |
| state         | active / idle / idle in transaction |
| query         | Current or last query               |
| query_start   | When query began                    |
| xact_start    | When transaction began              |
| backend_start | When session started                |

---

### Waiting & Blocking (Most Important)

| Column          | Meaning                     |
| --------------- | --------------------------- |
| wait_event_type | Lock / IO / LWLock / Client |
| wait_event      | Specific wait reason        |
| backend_xid     | Transaction ID              |
| backend_xmin    | Oldest visible XID          |

If `wait_event_type = 'Lock'` → **session is blocked**

---

## 4. Most Common Operational Queries

---

### 4.1 Find Active Queries

```sql
SELECT pid, usename, state, query
FROM pg_stat_activity
WHERE state = 'active';
```

---

### 4.2 Find Blocking Sessions

```sql
SELECT
  pid,
  wait_event_type,
  wait_event,
  query
FROM pg_stat_activity
WHERE wait_event_type = 'Lock';
```

---

### 4.3 Who Is Blocking Whom

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

---

### 4.4 Find Idle-in-Transaction (Very Dangerous)

```sql
SELECT pid, usename, xact_start, query
FROM pg_stat_activity
WHERE state = 'idle in transaction';
```

These sessions:

* Hold locks
* Prevent vacuum
* Cause production outages

 

### 4.5 Long-Running Transactions

```sql
SELECT pid, now() - xact_start AS tx_duration, query
FROM pg_stat_activity
WHERE xact_start IS NOT NULL
ORDER BY tx_duration DESC;
```



## 5. Killing a Problematic Session

### Cancel Query (Safe)

```sql
SELECT pg_cancel_backend(pid);
```

### Terminate Session (Force)

```sql
SELECT pg_terminate_backend(pid);
```

Use terminate only if cancel doesn’t work.



## 6. Real-World Examples

### Example 1: `SELECT FOR UPDATE` Blocking

You see:

* One session holding lock
* Multiple sessions waiting

`pg_stat_activity` immediately shows:

* Which query is blocking
* How long it has been running



### Example 2: Spring Batch Job Hung

Cause:

* Job crashed
* Transaction never committed

Diagnosis:

* `state = 'idle in transaction'`
* Holding locks on `BATCH_*` tables



### Example 3: API Latency Spike

Cause:

* One long UPDATE holding row lock
* All APIs waiting

Diagnosis:

* `wait_event_type = 'Lock'`

 

## 7. Difference vs Other System Views

| View               | Purpose                |
| ------------------ | ---------------------- |
| pg_stat_activity   | Live sessions          |
| pg_locks           | Low-level lock details |
| pg_stat_statements | Historical query stats |
| pg_stat_database   | DB-level metrics       |
 

## 8. Best Practices

* Always monitor `pg_stat_activity` in production
* Alert on `idle in transaction > N seconds`
* Set `application_name` per service
* Combine with `pg_locks` for deep analysis
* Use `lock_timeout` to avoid infinite waits

 

## 9. Mental Model

> `pg_stat_activity` answers **“What is happening right now in PostgreSQL?”**

If there is only **one PostgreSQL view you memorize**, this is it.
 

If you want, next I can:

* Explain `pg_locks` with examples
* Show a real deadlock trace
* Map this to Spring Boot connection pools
* Explain how PostgreSQL decides lock wait order

Just tell me.
