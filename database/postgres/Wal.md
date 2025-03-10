### **🔹 Write-Ahead Log (WAL) in PostgreSQL**  

**Write-Ahead Logging (WAL)** is a **crash recovery mechanism** in PostgreSQL that ensures **data integrity and durability**. It works by **logging all changes** before applying them to the actual database.

---

## **🔹 How WAL Works in PostgreSQL**
1. **Changes are first written to the WAL log** (before updating data files).  
2. **Only after logging**, changes are applied to the actual database.  
3. **If PostgreSQL crashes**, it can recover data from the WAL logs.  

This ensures that **no committed transaction is lost**, even in case of failure.

---

## **🔹 Why Use Write-Ahead Logging?**
✅ **Crash Recovery** → Ensures data safety after a system crash.  
✅ **Faster Writes** → Reduces disk I/O by batching updates.  
✅ **Replication & Streaming** → WAL helps in database replication and log shipping.  
✅ **Point-in-Time Recovery (PITR)** → Restore the database to a specific time.  

---

## **🔹 Example: WAL in Action**
Let's say you execute:
```sql
UPDATE employees SET salary = salary + 1000 WHERE id = 10;
```
**Steps in WAL:**
1️⃣ The change is **first recorded in the WAL log** (stored in `pg_wal/`).  
2️⃣ PostgreSQL **commits the transaction** once the WAL log is written.  
3️⃣ Later, background processes apply the changes to the actual database.  

---

## **🔹 WAL File Location in PostgreSQL**
- WAL files are stored in:  
  ```
  $PGDATA/pg_wal/
  ```
- You can list WAL files using:
  ```bash
  ls -lh $PGDATA/pg_wal/
  ```
- To check WAL status:
  ```sql
  SELECT * FROM pg_stat_wal;
  ```

---

## **🔹 Key Parameters for WAL Configuration**
You can configure WAL behavior in **`postgresql.conf`**:
```ini
wal_level = replica         # Sets logging level (minimal, replica, logical)
wal_buffers = 16MB          # Size of WAL buffers
checkpoint_timeout = 5min   # How often WAL is flushed
max_wal_size = 1GB          # Maximum WAL file size before flushing
```

---

## **🔹 Types of WAL Files**
1️⃣ **WAL Segment Files** → Contain logs of changes.  
2️⃣ **Checkpoint Records** → Periodically flush WAL data to the database.  
3️⃣ **Archived WAL Files** → Used for replication & backup.

---

## **🔹 WAL & Replication**
- **Streaming Replication**: Uses WAL logs to sync replicas.
- **Logical Replication**: Uses WAL to replicate specific tables.
- **Log Shipping**: Sends WAL archives to a standby server.

---

## **🔹 Summary**
| Feature       | Benefit |
|--------------|---------|
| **Crash Recovery** | Ensures no committed transaction is lost |
| **Faster Writes** | Writes changes only once (WAL first) |
| **Replication** | WAL enables database replication |
| **PITR (Backups)** | Restore data to a specific time |

Would you like a **detailed example on WAL-based recovery**?