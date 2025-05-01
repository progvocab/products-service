In Oracle databases, **table-level locks** are mechanisms used to control concurrent access to an entire table, ensuring data integrity when multiple users or sessions access or modify data simultaneously. Unlike row-level locks, which restrict access to specific rows, table-level locks apply to the entire table, impacting all rows. They are part of Oracle’s concurrency control system and can be acquired automatically during Data Manipulation Language (DML) operations or explicitly via the `LOCK TABLE` statement. Table-level locks are particularly useful for operations that affect a substantial portion of a table or require exclusive access, but they can reduce concurrency compared to row-level locks.

Below, I’ll explain table-level locks in Oracle, their modes, how they work, how to manage them, and provide examples. I’ll also integrate context from your previous questions (e.g., C++ programming) where relevant, such as discussing how a C++ application might interact with Oracle locks. The explanation incorporates insights from Oracle documentation and other reliable sources, critically examined to ensure accuracy.

---

### Key Concepts of Table-Level Locks
1. **Purpose**: Table-level locks prevent conflicting operations (e.g., updates, DDL changes) on a table during a transaction, ensuring data consistency in a multi-user environment.
2. **Granularity**: Locks the entire table, not individual rows, making it less granular than row-level locks.
3. **Automatic vs. Explicit**:
   - **Automatic**: Oracle applies table-level locks during DML operations (e.g., `INSERT`, `UPDATE`, `DELETE`) to prevent structural changes (e.g., table drops) during the operation.
   - **Explicit**: Users can issue the `LOCK TABLE` statement to manually lock a table in a specific mode.
3. **Concurrency Impact**: Table-level locks can reduce throughput and concurrency, especially in high-transaction environments, as they restrict access to the entire table.
4. **No Lock Escalation**: Oracle does not escalate row-level locks to table-level locks, unlike some other databases. This avoids deadlocks but may increase lock overhead for large updates.[](https://www.oraclemasterpiece.com/2021/01/oracle-locking-mechanism/)

---

### Table-Level Lock Modes
Oracle defines several table-level lock modes, each with different levels of restrictiveness and concurrency. These modes determine what operations other sessions can perform on the locked table. The modes, from least to most restrictive, are:

1. **Row Share (RS, also called SHARE UPDATE)**:
   - **Purpose**: Allows concurrent access to the table but prevents other sessions from locking the entire table in exclusive mode.
   - **Use Case**: Acquired automatically during `SELECT ... FOR UPDATE` to lock specific rows while allowing other DML operations.
   - **Concurrency**: High; other sessions can read and modify different rows.
   - **Example**: Prevents DDL operations like `DROP TABLE` during a transaction.

2. **Row Exclusive (RX)**:
   - **Purpose**: Similar to Row Share but also prevents locking the table in SHARE mode.
   - **Use Case**: Automatically acquired during DML operations (`INSERT`, `UPDATE`, `DELETE`).
   - **Concurrency**: High; other sessions can perform DML on different rows but cannot lock the table in SHARE mode.
   - **Example**: Ensures no shared locks interfere during updates.

3. **Share (S)**:
   - **Purpose**: Allows concurrent queries but prevents updates to the table.
   - **Use Case**: Useful for read-heavy operations where the table’s data should remain unchanged.
   - **Concurrency**: Moderate; multiple sessions can acquire SHARE locks, but no DML is allowed.
   - **Example**: Protects a table during a complex report generation.

4. **Share Row Exclusive (SRX)**:
   - **Purpose**: Allows reading rows but prevents updates and locking in SHARE mode.
   - **Use Case**: Used when a session needs to read the entire table and ensure no updates or SHARE locks occur.
   - **Concurrency**: Lower; restricts updates and SHARE locks.
   - **Example**: Preparing a table for a bulk read operation.

5. **Exclusive (X)**:
   - **Purpose**: Grants exclusive write access, allowing only queries (`SELECT`) by other sessions and prohibiting all other operations.
   - **Use Case**: Used for operations requiring complete control, like bulk updates or DDL (e.g., `ALTER TABLE`).
   - **Concurrency**: Lowest; only one session can hold an Exclusive lock, blocking most operations.
   - **Example**: Locking a table during a schema change.

**Note**: When locking partitions or subpartitions, an implicit table-level lock is acquired. For example, a SHARE lock on a subpartition results in a ROW SHARE lock on the table, and an EXCLUSIVE lock results in a ROW EXCLUSIVE lock.[](https://docs.oracle.com/en/database/oracle/oracle-database/23/sqlrf/LOCK-TABLE.html)

---

### How Table-Level Locks Work
- **Automatic Locking**: Oracle automatically acquires table-level locks during DML operations to prevent DDL conflicts. For example:
  - An `UPDATE` statement acquires a Row Exclusive (RX) table lock and row-level locks (TX) on modified rows.[](https://docs.oracle.com/en/database/oracle/oracle-database/23/sqlrf/Automatic-Locks-in-DML-Operations.html)
  - The table lock ensures the table’s structure isn’t altered (e.g., dropped) during the transaction.
- **Explicit Locking**: The `LOCK TABLE` statement allows users to override Oracle’s automatic locking for specific needs:
  ```sql
  LOCK TABLE table_name IN lock_mode MODE [NOWAIT | WAIT seconds];
  ```
  - **NOWAIT**: Returns immediately if the lock cannot be acquired, raising an error (e.g., `ORA-00054: resource busy`).[](https://docs.oracle.com/en/database/oracle/oracle-database/23/sqlrf/LOCK-TABLE.html)
  - **WAIT**: Waits for the specified seconds or indefinitely (if omitted) to acquire the lock.
- **Lock Storage**: Oracle stores lock information in the data block containing the locked resource, not in a centralized lock manager, improving efficiency.[](https://docs.oracle.com/en/database/oracle/oracle-database/23/sqlrf/Automatic-Locks-in-DML-Operations.html)
- **Lock Release**: Locks are held until the transaction commits (`COMMIT`) or rolls back (`ROLLBACK`).[](https://www.studytonight.com/plsql/locks-in-plsql)

---

### Syntax for Explicit Table Locking
```sql
LOCK TABLE [schema.]table_name [PARTITION (partition_name) | SUBPARTITION (subpartition_name)]
IN {ROW SHARE | ROW EXCLUSIVE | SHARE | SHARE ROW EXCLUSIVE | EXCLUSIVE} MODE
[NOWAIT | WAIT integer];
```
- **Privileges**: You need to own the table, have the `LOCK ANY TABLE` system privilege, or have an object privilege (except `READ`) on the table.[](https://docs.oracle.com/en/database/oracle/oracle-database/23/sqlrf/LOCK-TABLE.html)
- **Remote Tables**: Tables on a remote database can be locked via a database link, but all tables in a single `LOCK TABLE` statement must be on the same database.[](https://docs.oracle.com/en/database/oracle/oracle-database/23/sqlrf/LOCK-TABLE.html)

**Example**:
```sql
LOCK TABLE employees IN EXCLUSIVE MODE NOWAIT;
```
This locks the `employees` table in Exclusive mode, failing immediately if another session holds a conflicting lock.

---

### When to Use Table-Level Locks
Table-level locks are useful in specific scenarios:
1. **Bulk Operations**: When a transaction modifies most rows in a table, a table-level lock can be more efficient than multiple row-level locks.[](https://docs.oracle.com/en/database/other-databases/timesten/22.1/introduction/locking-levels.html)
2. **DDL Operations**: To prevent DML operations during structural changes (e.g., `ALTER TABLE`).
3. **Data Consistency**: To ensure no changes occur during a critical read operation (e.g., generating a report).
4. **Low Concurrency Needs**: When high concurrency isn’t required, such as during batch processing or database initialization.[](https://docs.oracle.com/en/database/other-databases/timesten/22.1/introduction/locking-levels.html)

However, table-level locks should be used cautiously because they reduce concurrency, especially in high-transaction systems. Row-level locking is generally preferred for better throughput.[](https://docs.oracle.com/en/database/other-databases/timesten/22.1/introduction/locking-levels.html)

---

### Checking for Table-Level Locks
To identify table-level locks, query the following views:
1. **`V$LOCKED_OBJECT`**:
   - Lists locked objects, including tables, with session details.
   ```sql
   SELECT b.owner, b.object_name, a.oracle_username, a.os_user_name
   FROM v$locked_object a
   JOIN dba_objects b ON a.object_id = b.object_id
   WHERE b.object_name = 'MYTABLE_NAME';
   ```
  [](https://www.quora.com/How-can-you-check-if-a-table-is-locked-in-Oracle)

2. **`V$LOCK`**:
   - Provides detailed lock information, including blocking sessions.
   ```sql
   SELECT l.sid, l.type, l.lmode, l.request, o.object_name
   FROM v$lock l
   JOIN dba_objects o ON l.id1 = o.object_id
   WHERE o.object_name = 'MYTABLE_NAME';
   ```
  [](https://medium.com/%40Radwan.salameh/identifying-and-removing-locks-on-oracle-tables-2df7027d0dea)

3. **`V$SESSION` and `GV$LOCK`** (for RAC):
   - Identify blocking sessions and their details.
   ```sql
   SELECT s.sid, s.serial#, s.username, s.machine
   FROM v$session s
   JOIN v$locked_object lo ON s.sid = lo.session_id
   JOIN dba_objects o ON lo.object_id = o.object_id
   WHERE o.object_name = 'MYTABLE_NAME';
   ```
  [](https://orahow.com/find-and-remove-table-lock-in-oracle/)

**Note**: Oracle does not store historical lock data, so you can only check current locks. For historical analysis, use `V$ACTIVE_SESSION_HISTORY` (requires licensing).[](https://dba.stackexchange.com/questions/56615/how-to-determine-if-an-oracle-table-is-locked-or-not)

---

### Removing Table-Level Locks
To release a table-level lock, you can:
1. **Commit or Rollback**: The session holding the lock must issue `COMMIT` or `ROLLBACK`.
2. **Kill the Session**: If the session is unresponsive, a DBA can kill it:
   ```sql
   SELECT s.sid, s.serial#, s.process
   FROM v$session s
   JOIN v$locked_object lo ON s.sid = lo.session_id
   JOIN dba_objects o ON lo.object_id = o.object_id
   WHERE o.object_name = 'MYTABLE_NAME';

   ALTER SYSTEM KILL SESSION 'sid,serial#' IMMEDIATE;
   ```
   Example:
   ```sql
   ALTER SYSTEM KILL SESSION '49,45509' IMMEDIATE;
   ```
  [](https://www.dbgenre.com/post/how-to-find-and-remove-table-level-lock-in-oracle)

3. **Truncate as Alternative**: If a table is locked due to a large `DELETE` operation, truncating the table (DDL) is faster than deleting rows (DML) and resets the High Water Mark. However, DDL operations may fail if a lock exists:
   ```sql
   ALTER SESSION SET ddl_lock_timeout=30;
   TRUNCATE TABLE OWNER.MYTABLE_NAME DROP STORAGE;
   ```
   If this fails with `ORA-00054`, kill the locking session first.[](https://www.dbgenre.com/post/how-to-find-and-remove-table-level-lock-in-oracle)

---

### Code Example: Table-Level Locks in a C++ Application
Since you’ve asked about C++ (e.g., `std::vector`, `{fmt}`), here’s how a C++ application using Oracle’s OCCI (Oracle C++ Call Interface) might interact with table-level locks, incorporating `{fmt}` for formatting output:

```cpp
#include <occi.h>
#include <fmt/core.h>
#include <iostream>
#include <string>

int main() {
    oracle::occi::Environment* env = oracle::occi::Environment::createEnvironment();
    std::string user = "hr", pass = "password", connStr = "//localhost:1521/orcl";

    try {
        oracle::occi::Connection* conn = env->createConnection(user, pass, connStr);
        oracle::occi::Statement* stmt = conn->createStatement();

        // Lock the employees table in EXCLUSIVE mode
        stmt->setSQL("LOCK TABLE employees IN EXCLUSIVE MODE NOWAIT");
        try {
            stmt->executeUpdate();
            fmt::print("Table employees locked in EXCLUSIVE mode\n");
        } catch (oracle::occi::SQLException& e) {
            fmt::print("Error locking table: {}\n", e.what());
            if (e.getErrorCode() == 54) {  // ORA-00054: resource busy
                fmt::print("Table is already locked by another session\n");
            }
        }

        // Perform operations (e.g., bulk update)
        stmt->setSQL("UPDATE employees SET salary = salary * 1.1");
        int rowsUpdated = stmt->executeUpdate();
        fmt::print("Updated {} rows\n", rowsUpdated);

        // Commit to release the lock
        conn->commit();
        fmt::print("Transaction committed, lock released\n");

        env->terminateConnection(conn);
    } catch (oracle::occi::SQLException& e) {
        fmt::print("Database error: {}\n", e.what());
    }

    oracle::occi::Environment::terminateEnvironment(env);
    return 0;
}
```
- **Compile**: Link with OCCI and `{fmt}` libraries (e.g., `-locci -lfmt`).
- **Output**:
  ```
  Table employees locked in EXCLUSIVE mode
  Updated 107 rows
  Transaction committed, lock released
  ```
  Or, if the table is locked:
  ```
  Error locking table: ORA-00054: resource busy and acquire with NOWAIT specified
  Table is already locked by another session
  ```
- **Explanation**:
  - The C++ program connects to an Oracle database, attempts to lock the `employees` table, performs an update, and commits to release the lock.
  - `{fmt}` is used to format error messages and status updates cleanly.
  - If another session holds a conflicting lock, `ORA-00054` is caught and handled.

---

### Example: Explicit Table Lock in SQL
```sql
-- Session 1: Lock table in EXCLUSIVE mode
BEGIN
    LOCK TABLE employees IN EXCLUSIVE MODE NOWAIT;
    UPDATE employees SET salary = salary + 1000;
    -- Simulate a long-running transaction
    DBMS_LOCK.SLEEP(30);
    COMMIT;
END;
/

-- Session 2: Attempt to lock the same table
LOCK TABLE employees IN SHARE MODE NOWAIT;
-- Fails with: ORA-00054: resource busy and acquire with NOWAIT specified

-- Session 2: Check for locks
SELECT b.owner, b.object_name, a.oracle_username, a.sid
FROM v$locked_object a
JOIN dba_objects b ON a.object_id = b.object_id
WHERE b.object_name = 'EMPLOYEES';

-- Session 2: Kill the blocking session (DBA only)
ALTER SYSTEM KILL SESSION 'sid,serial#' IMMEDIATE;
```
- **Explanation**:
  - Session 1 locks the table and performs an update, holding the lock for 30 seconds.
  - Session 2 fails to acquire a SHARE lock due to the EXCLUSIVE lock.
  - Session 2 queries `V$LOCKED_OBJECT` to identify the blocking session and kills it (if privileged).

---

### Strategies to Minimize Table-Level Locks
To reduce contention and avoid table-level locks:
1. **Use Row-Level Locking**: Prefer `SELECT ... FOR UPDATE` for small, targeted updates to maximize concurrency.[](https://docs.oracle.com/en/database/other-databases/timesten/22.1/introduction/locking-levels.html)
2. **Optimize Transactions**: Keep transactions short to release locks quickly.
3. **Partition Tables**: Lock specific partitions instead of the entire table for large datasets.[](https://www.process.st/how-to/check-table-lock-in-oracle/)
4. **Use DDL Alternatives**: For operations like deleting all rows, use `TRUNCATE` instead of `DELETE` to avoid DML locks.[](https://www.dbgenre.com/post/how-to-find-and-remove-table-level-lock-in-oracle)
5. **Monitor Locks**: Regularly check `V$LOCKED_OBJECT` and `V$SESSION` to detect and resolve lock conflicts.[](https://www.process.st/how-to/check-table-lock-in-oracle/)
6. **Set Isolation Levels**: Use `READ COMMITTED` (default) instead of `SERIALIZABLE` to reduce lock duration, unless strict consistency is needed.[](https://docs.oracle.com/en/database/other-databases/timesten/22.1/introduction/locking-levels.html)

---

### Common Issues and Solutions
1. **ORA-00054: Resource Busy**:
   - **Cause**: Another session holds a conflicting lock.
   - **Solution**: Use `NOWAIT` or `WAIT` clauses, check `V$LOCKED_OBJECT`, and kill the blocking session if necessary.[](https://www.dbgenre.com/post/how-to-find-and-remove-table-level-lock-in-oracle)
2. **Deadlocks**:
   - **Cause**: Two sessions wait for each other’s locks.
   - **Solution**: Oracle detects deadlocks and rolls back one transaction. Design applications to acquire locks in a consistent order.
3. **Performance Bottlenecks**:
   - **Cause**: Excessive table-level locks in high-concurrency systems.
   - **Solution**: Favor row-level locks, partition tables, and optimize queries with indexes.[](https://www.process.st/how-to/check-table-lock-in-oracle/)

---

### Comparison with Other Databases
- **Oracle vs. Others**: Unlike some databases (e.g., SQL Server), Oracle does not escalate row locks to table locks, reducing deadlock risks but potentially increasing lock overhead.[](https://www.oraclemasterpiece.com/2021/01/oracle-locking-mechanism/)
- **MySQL**: Uses `LOCK TABLES` for explicit locking, which is less granular and lacks Oracle’s automatic DML locking sophistication.[](https://docs.oracle.com/cd/E17952_01/mysql-5.7-en/lock-tables.html)
- **PostgreSQL**: Relies heavily on row-level locks and MVCC, with table-level locks for DDL, similar to Oracle but with different implementation details.

---

### Conclusion
Table-level locks in Oracle are a powerful tool for controlling access to entire tables, used automatically in DML operations or explicitly via `LOCK TABLE`. They come in various modes (Row Share, Row Exclusive, Share, Share Row Exclusive, Exclusive), each balancing concurrency and restrictiveness. While useful for bulk operations or DDL, they can reduce concurrency, so row-level locks are preferred in high-transaction environments. Monitoring and managing locks through views like `V$LOCKED_OBJECT` and `V$LOCK` is crucial for performance and conflict resolution. In a C++ context, applications can interact with these locks using OCCI, with `{fmt}` enhancing error handling and logging.

If you need more details (e.g., specific lock scenarios, integration with C++ applications, or RAC environments), let me know!