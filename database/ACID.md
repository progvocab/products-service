### **ACID Properties in PostgreSQL with Code Examples**  

**ACID** stands for **Atomicity, Consistency, Isolation, and Durability**. These properties ensure that **database transactions** are reliable even in cases of system failures, crashes, or concurrent access.

---

## **1. ACID Properties in PostgreSQL**  

| **Property**  | **Description** |
|--------------|----------------|
| **Atomicity** | A transaction is either **fully completed** or **fully rolled back**. |
| **Consistency** | Ensures the database remains in a **valid state** before and after a transaction. |
| **Isolation** | Transactions are executed **independently** without interference. |
| **Durability** | Once a transaction is **committed**, it is permanently stored in the database. |

---

## **2. Understanding ACID with PostgreSQL Examples**  

### **A. Atomicity Example** (Rollback on Failure)  
Atomicity ensures that **all operations in a transaction succeed or none are applied**.  

#### **Example: Transfer Money Between Accounts**  
```sql
BEGIN;  -- Start Transaction

UPDATE accounts SET balance = balance - 100 WHERE account_id = 1;  -- Debit
UPDATE accounts SET balance = balance + 100 WHERE account_id = 2;  -- Credit

COMMIT;  -- Commit Transaction
```
✅ If all statements succeed, the transaction is **committed**.  
❌ If any statement fails (e.g., insufficient balance), we **ROLLBACK**:  

```sql
BEGIN;

UPDATE accounts SET balance = balance - 100 WHERE account_id = 1;
UPDATE accounts SET balance = balance + 100 WHERE account_id = 2;

-- Simulate failure
ROLLBACK;  -- Undo the transaction
```
🔹 This ensures **no partial updates** occur (Atomicity).  

---

### **B. Consistency Example** (Maintain Data Integrity)  
Consistency ensures that **only valid data is stored** in the database.

#### **Example: Enforce a Business Rule Using Constraints**
```sql
CREATE TABLE accounts (
    account_id SERIAL PRIMARY KEY,
    balance DECIMAL CHECK (balance >= 0)  -- Ensures balance can't be negative
);
```
Now, if we try:
```sql
INSERT INTO accounts (balance) VALUES (-50);
```
❌ **ERROR:**  
```
ERROR:  new row for relation "accounts" violates check constraint "accounts_balance_check"
```
✅ This prevents **invalid data**, maintaining database **consistency**.

---

### **C. Isolation Example** (Prevent Concurrent Issues)  
Isolation ensures that **concurrent transactions do not interfere with each other**. PostgreSQL supports **four isolation levels**:

| **Isolation Level**      | **Effect** |
|------------------------|---------------------------|
| Read Uncommitted      | Can see uncommitted changes (Not supported in PostgreSQL). |
| Read Committed        | Default level; sees only committed data. |
| Repeatable Read       | Ensures consistent snapshots of data. |
| Serializable          | Fully isolated transactions (strictest level). |

#### **Example: Isolation Using `READ COMMITTED` (Default in PostgreSQL)**  

**Session 1: Start Transaction (Not Committed Yet)**  
```sql
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE account_id = 1;
```

**Session 2: Try Reading Before Commit**  
```sql
SELECT balance FROM accounts WHERE account_id = 1;
```
✅ **Output (Old balance is shown because transaction is not committed yet)**  
```
 balance 
---------
 500
(1 row)
```

**Session 1: Commit the Transaction**  
```sql
COMMIT;
```
🔹 Now, **Session 2** will see the **updated balance** (ensuring isolation).

---

### **D. Durability Example** (Data Persistence)  
Durability ensures that **once a transaction is committed, the changes persist even after system failures**.

#### **Example: PostgreSQL WAL (Write-Ahead Logging) for Durability**  
PostgreSQL uses **WAL** to store committed transactions permanently.

```sql
SHOW wal_level;
```
✅ Output:  
```
 wal_level 
------------
 replica
(1 row)
```
#### **Steps to Ensure Durability**
1. PostgreSQL writes transactions to WAL logs.  
2. The WAL logs are flushed to disk.  
3. The data is permanently stored after `COMMIT;`.  

Even if the **server crashes**, committed data is **not lost**.

---

## **3. Summary of ACID in PostgreSQL**  

| **Property**  | **Ensured By** |
|--------------|---------------|
| **Atomicity** | Transactions (`BEGIN`, `COMMIT`, `ROLLBACK`) |
| **Consistency** | Constraints (`CHECK`, `FOREIGN KEY`, `UNIQUE`) |
| **Isolation** | Isolation levels (`READ COMMITTED`, `SERIALIZABLE`) |
| **Durability** | WAL (Write-Ahead Logging), `COMMIT` |

---

## **4. Conclusion**  
PostgreSQL ensures **ACID compliance** through transactions, constraints, isolation levels, and WAL logging. These properties make it **highly reliable for banking, e-commerce, and enterprise applications**.

Would you like a **real-world ACID example using Golang & PostgreSQL**? 🚀