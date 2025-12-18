`MERGE INTO` is a **single SQL statement used to synchronize data between two datasets** by doing:

* **INSERT**
* **UPDATE**
* **DELETE**

in **one atomic operation**, based on a **matching condition**.
 

> **“If the row exists → update it, if it doesn’t → insert it (optionally delete)”**

* `MERGE INTO` is for **data synchronization**
* Combines **INSERT + UPDATE + DELETE**
* Atomic and concurrency-safe
* Best for **batch and staging workflows**
* Available in **PostgreSQL 15+**

### Why `MERGE INTO` Exists

Before `MERGE`, you had to write **multiple statements**:

```sql
UPDATE ...
INSERT ...
DELETE ...
```

Problems:

* Race conditions
* Complex logic
* Multiple scans
* Not atomic

`MERGE INTO` solves all of this in **one statement**.
 

### Basic Syntax (PostgreSQL 15+)

```sql
MERGE INTO target_table t
USING source_table s
ON t.id = s.id

WHEN MATCHED THEN
  UPDATE SET col1 = s.col1

WHEN NOT MATCHED THEN
  INSERT (id, col1)
  VALUES (s.id, s.col1);
```
 

### What Happens Internally

1. PostgreSQL joins **target** and **source**
2. For each row:

   * If `ON` condition matches → `WHEN MATCHED`
   * If no match → `WHEN NOT MATCHED`
3. Executes **exactly one action per row**
4. Entire statement is **atomic**

---

## 4. Real-World Example (Upsert)

### Sync staging orders into main orders table

```sql
MERGE INTO orders o
USING orders_staging s
ON o.order_id = s.order_id

WHEN MATCHED THEN
  UPDATE SET
    status = s.status,
    amount = s.amount

WHEN NOT MATCHED THEN
  INSERT (order_id, status, amount)
  VALUES (s.order_id, s.status, s.amount);
```


## 5. MERGE vs INSERT … ON CONFLICT

| Feature             | MERGE         | INSERT ON CONFLICT  |
| ------------------- | ------------- | ------------------- |
| Source              | Table / Query | VALUES only         |
| UPDATE + INSERT     | Yes           | Yes                 |
| DELETE              | Yes           | No                  |
| Multiple conditions | Yes           | No                  |
| ANSI SQL            | Yes           | PostgreSQL-specific |
| Batch sync          | ✅ Best        | ❌ Limited           |

Use:

* **ON CONFLICT** → simple upsert
* **MERGE** → data synchronization

---

## 6. DELETE with MERGE

```sql
MERGE INTO orders o
USING orders_staging s
ON o.order_id = s.order_id

WHEN MATCHED AND s.is_active = false THEN
  DELETE;
```

Very useful for:

* Soft deletes
* Data reconciliation

---

## 7. Locking Behavior (Important)

* MERGE acquires **row-level locks**
* Behaves like:

  * `UPDATE` for matched rows
  * `INSERT` for new rows
* Competing updates **wait**, not fail
* Can block under high concurrency

👉 Same rules as `SELECT FOR UPDATE`

---

## 8. Transaction Safety

* Fully transactional
* Rolls back entirely on error
* Safe for concurrent execution
* Honors constraints and triggers

---

## 9. Common Use Cases

* Data warehouse loads
* CDC (Change Data Capture)
* Sync between staging → prod tables
* Batch reconciliation
* Replacing complex upsert logic
* Spring Batch final load step

---

## 10. PostgreSQL Version Support

| Version        | Support         |
| -------------- | --------------- |
| PostgreSQL ≤14 | ❌ Not available |
| PostgreSQL 15+ | ✅ Supported     |

---

## 11. Performance Considerations

* Index on `ON` condition is **mandatory**
* Large MERGE without index = full scan
* Avoid huge transactions
* Batch source data if very large

---

## 12. Common Mistakes

* Missing index on join key
* Expecting MERGE to be faster than UPDATE always
* Using MERGE for OLTP hot rows (can cause contention)
* Forgetting constraints can still fail the MERGE

---

## 13. Simple Mental Model

> **MERGE = JOIN + IF/ELSE + DML + TRANSACTION**

---

## 14. When NOT to Use MERGE

* High-frequency OLTP updates on hot rows
* Simple single-row upserts (`ON CONFLICT` is better)
* When source is a single VALUES clause
 
 


 
More :

* Compare MERGE vs `SELECT FOR UPDATE` patterns
* Show MERGE deadlock scenarios
* Explain MERGE with Spring Batch
* Provide performance tuning checklist
 
