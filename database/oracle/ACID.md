ACID is a set of properties that ensure reliable processing of database transactions. Let's break down **ACID** in the context of **Oracle Database**, along with **code examples**, **design patterns**, **data structures**, and **use cases** relevant to Oracle.

---

## 🔸 What is ACID?

ACID stands for:

| Property    | Description |
|-------------|-------------|
| **A**tomicity | Ensures that all parts of a transaction are completed; if one part fails, the whole transaction is rolled back. |
| **C**onsistency | Ensures the database remains in a valid state before and after a transaction. |
| **I**solation | Ensures transactions occur independently, preventing dirty reads or phantom reads. |
| **D**urability | Ensures that committed data is saved permanently, even in case of system failure. |

---

## 🔸 Oracle Implementation of ACID

### ✅ 1. Atomicity in Oracle

Oracle ensures atomicity using `SAVEPOINT`, `ROLLBACK`, and `COMMIT`.

**Example:**
```sql
BEGIN
  INSERT INTO employees (id, name) VALUES (1, 'Alice');
  INSERT INTO departments (id, name) VALUES (1, 'HR');

  -- if something goes wrong
  -- ROLLBACK;

  COMMIT;
END;
```

**If one insert fails**, the whole transaction can be rolled back.

---

### ✅ 2. Consistency in Oracle

Oracle ensures that constraints (PK, FK, CHECK, NOT NULL) and triggers maintain consistency.

**Example:**
```sql
CREATE TABLE employees (
  id NUMBER PRIMARY KEY,
  name VARCHAR2(50) NOT NULL,
  dept_id NUMBER,
  CONSTRAINT fk_dept FOREIGN KEY (dept_id) REFERENCES departments(id)
);
```

If a transaction violates a foreign key or other constraint, Oracle rolls it back.

---

### ✅ 3. Isolation in Oracle

Oracle provides **multiple isolation levels**:

| Isolation Level         | Description |
|-------------------------|-------------|
| READ COMMITTED (default)| Prevents dirty reads. |
| SERIALIZABLE            | Transactions are fully isolated. |
| READ ONLY               | Read-only snapshot of the DB. |

**Set isolation level:**
```sql
SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;
```

---

### ✅ 4. Durability in Oracle

Once a transaction is committed, Oracle writes to redo logs and datafiles.

**Mechanism:**
- Oracle **writes to redo logs**
- **LGWR (Log Writer)** flushes these logs to disk
- Even if a crash occurs, redo logs are used for recovery

---

## 🔸 Code Example – Full ACID Transaction

```sql
BEGIN
  SAVEPOINT before_insert;

  INSERT INTO accounts (id, balance) VALUES (101, 5000);
  INSERT INTO transactions (account_id, amount, type) VALUES (101, 5000, 'CREDIT');

  IF SQL%ROWCOUNT < 2 THEN
    ROLLBACK TO before_insert;
  END IF;

  COMMIT;
END;
```

---

## 🔸 Design Patterns with Oracle and ACID

| Pattern | Usage |
|--------|--------|
| **Unit of Work** | Treat multiple DB operations as a single transaction. |
| **Repository** | Abstracts and encapsulates all data access logic. |
| **Transaction Script** | Procedural logic that organizes transactions. |

**Example (PL/SQL Transaction Script):**
```sql
CREATE OR REPLACE PROCEDURE transfer_funds(p_from NUMBER, p_to NUMBER, p_amount NUMBER) AS
BEGIN
  UPDATE accounts SET balance = balance - p_amount WHERE id = p_from;
  UPDATE accounts SET balance = balance + p_amount WHERE id = p_to;

  COMMIT;
EXCEPTION
  WHEN OTHERS THEN
    ROLLBACK;
    RAISE;
END;
```

---

## 🔸 Data Structures Oracle Uses for ACID

| Structure | Role |
|----------|------|
| **Redo Logs** | Ensure **Durability** – used to recover committed transactions. |
| **Undo Segments** | Ensure **Atomicity** and **Isolation** – rollback uncommitted changes. |
| **System Change Number (SCN)** | Ensures **Consistency** – tracks transaction states. |
| **Locks & Latches** | Ensure **Isolation** – prevent race conditions or dirty reads. |

---

## 🔸 Use Cases for ACID in Oracle

| Use Case | Why ACID Matters |
|----------|------------------|
| **Banking Systems** | Transactions must be atomic and consistent (transfer funds). |
| **Inventory Management** | Prevent double shipping or incorrect stock. |
| **Payroll Processing** | Ensures accurate calculation and disbursement. |
| **Booking Systems (Airlines/Hotels)** | Avoid overbooking, maintain consistency. |
| **E-commerce Transactions** | Order placement, payment, inventory update must be atomic. |

---

## 🔸 Bonus: ACID Test Query

Here’s a test to verify atomic rollback:

```sql
BEGIN
  INSERT INTO employees (id, name) VALUES (2, 'Bob');
  INSERT INTO employees (id, name) VALUES (2, 'Bob'); -- duplicate PK

  COMMIT;
EXCEPTION
  WHEN DUP_VAL_ON_INDEX THEN
    DBMS_OUTPUT.PUT_LINE('Rolling back due to duplicate entry');
    ROLLBACK;
END;
```

---

Would you like a Java + Spring example of ACID transactions using Oracle JDBC or Spring Data JPA?
