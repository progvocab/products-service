Here’s a comprehensive explanation of **database anomalies** related to **concurrent transactions** with **query examples**, including how they relate to **isolation levels**:

---

### **1. Dirty Read**
**Definition:** A transaction reads data written by another uncommitted transaction.

**Occurs In:** `READ UNCOMMITTED` (Not supported in Oracle)

**Example:**
```sql
-- Session A
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
-- No COMMIT yet

-- Session B
SELECT balance FROM accounts WHERE id = 1; -- Reads uncommitted data (dirty read)
```

**Problem:** If Session A rolls back, Session B has read invalid data.

---

### **2. Non-Repeatable Read**
**Definition:** A transaction reads the same row twice and gets different results because another transaction modified it in the meantime.

**Occurs In:** `READ COMMITTED` (Oracle default)

**Example:**
```sql
-- Session A
BEGIN;
SELECT balance FROM accounts WHERE id = 1; -- Reads 1000

-- Session B
UPDATE accounts SET balance = 1200 WHERE id = 1;
COMMIT;

-- Session A
SELECT balance FROM accounts WHERE id = 1; -- Reads 1200 (different from earlier)
```

**Problem:** Session A sees **inconsistent reads** of the same row.

---

### **3. Phantom Read**
**Definition:** A transaction re-executes a query and finds new rows added or removed by another transaction.

**Occurs In:** `READ COMMITTED`, `REPEATABLE READ` (but **not** in `SERIALIZABLE`)

**Example:**
```sql
-- Session A
BEGIN;
SELECT * FROM orders WHERE amount > 100; -- Returns 5 rows

-- Session B
INSERT INTO orders (id, amount) VALUES (101, 150);
COMMIT;

-- Session A
SELECT * FROM orders WHERE amount > 100; -- Now returns 6 rows (phantom row)
```

**Problem:** Session A sees a **new row** that wasn't there earlier.

---

### **4. Lost Update**
**Definition:** Two transactions read the same row and update it based on the read value, overwriting each other’s changes.

**Occurs In:** `READ COMMITTED` without proper locking

**Example:**
```sql
-- Session A
SELECT balance FROM accounts WHERE id = 1; -- 1000

-- Session B
SELECT balance FROM accounts WHERE id = 1; -- 1000

-- Session A
UPDATE accounts SET balance = 1100 WHERE id = 1;

-- Session B
UPDATE accounts SET balance = 1200 WHERE id = 1;
```

**Problem:** Update by Session A is **lost** when Session B commits later.

---

### **5. Uncommitted Dependency**
Alias for **Dirty Read** — when a transaction reads data that has not yet been committed.

---

### **6. Inconsistent Analysis (or Non-Atomic View)**
**Definition:** A transaction reads data across multiple rows that another transaction is modifying simultaneously.

**Example:**
```sql
-- Session A: wants to calculate total balance
SELECT SUM(balance) FROM accounts;

-- Session B: Moves $100 from account 1 to account 2
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
UPDATE accounts SET balance = balance + 100 WHERE id = 2;
COMMIT;

-- Session A: May read one change but not the other => inconsistent total
```

---

### **Isolation Levels Summary:**

| Isolation Level     | Dirty Read | Non-Repeatable Read | Phantom Read |
|---------------------|------------|----------------------|---------------|
| Read Uncommitted    | Yes        | Yes                  | Yes           |
| Read Committed      | No         | Yes                  | Yes           |
| Repeatable Read     | No         | No                   | Yes           |
| Serializable        | No         | No                   | No            |

---

Would you like equivalent Spring Data or JDBC examples for these isolation scenarios?