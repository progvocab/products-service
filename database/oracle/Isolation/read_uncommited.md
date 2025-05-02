### **Creating a Transaction in Oracle**

Oracle automatically starts a **transaction** when any **DML (Data Manipulation Language)** statement like `INSERT`, `UPDATE`, or `DELETE` is issued. The transaction ends when you explicitly call `COMMIT` or `ROLLBACK`.

---

### **1. Basic Transaction Example**

```sql
-- Transaction begins implicitly
UPDATE employees SET salary = salary + 1000 WHERE employee_id = 101;

-- Not visible to other sessions until committed
COMMIT; -- Makes changes permanent
-- or
ROLLBACK; -- Undo all changes in this transaction
```

You can wrap multiple statements within one transaction:

```sql
BEGIN
  UPDATE employees SET salary = salary + 1000 WHERE department_id = 10;
  DELETE FROM employees WHERE employee_id = 105;
  INSERT INTO audit_log VALUES ('Salary updated and employee deleted', SYSDATE);
  COMMIT;
END;
```

---

### **2. Read Uncommitted Data in Oracle**

Oracle **does not support** reading **uncommitted data** directly, unlike databases like SQL Server with `READ UNCOMMITTED` isolation.

Oracle uses **multi-version concurrency control (MVCC)** and **Serializable/Read Committed** isolation by default, which ensures:

- You can **never read uncommitted changes** made by another transaction.
- Instead, Oracle provides **read consistency** (returns a consistent snapshot).

---

### **Alternative (But Risky) Method: Using `DBMS_LOCK` for Dirty Reads (not recommended)**

You **can’t read uncommitted data** unless you **write custom code**, like:
- Polling temp tables
- Using log tables for outbox patterns
- Or uncommitted messages outside Oracle

---

### **Isolation Levels in Oracle**

| Isolation Level      | Can Read Uncommitted? | Notes                            |
|----------------------|------------------------|----------------------------------|
| **READ COMMITTED**   | No                     | Default; reads committed data    |
| **SERIALIZABLE**     | No                     | Reads consistent snapshot        |
| **READ ONLY**        | No                     | Like serializable, but read-only|

---

### **Conclusion**

- Oracle **transactions begin automatically** on DML.
- Use `COMMIT` / `ROLLBACK` to finish.
- Oracle **never allows reading uncommitted data**.
- Use **logging or events** if partial visibility is needed.

Would you like an example comparing Oracle's isolation level behavior with another database like PostgreSQL or SQL Server?