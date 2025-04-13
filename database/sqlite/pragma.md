In **databases**, particularly in **SQLite**, a `PRAGMA` statement is a special command used to **query or modify the internal behavior of the database engine**. It's similar to SQL, but non-standard (not part of ANSI SQL), and usually specific to the database engine.

### SQLite `PRAGMA` — Overview

A `PRAGMA` in SQLite allows developers to:
- Adjust performance settings
- Inspect or change internal schema details
- Enable debugging features
- Modify locking, journal modes, and caching

---

### Common SQLite `PRAGMA` Statements

| PRAGMA Command                    | Description                                                                 |
|----------------------------------|-----------------------------------------------------------------------------|
| `PRAGMA table_info(table_name);` | Returns info about the columns in a table (name, type, etc.)               |
| `PRAGMA foreign_keys = ON;`      | Enforces foreign key constraints                                            |
| `PRAGMA journal_mode = WAL;`     | Changes journal mode (e.g., `DELETE`, `WAL`, `TRUNCATE`)                   |
| `PRAGMA synchronous = NORMAL;`   | Adjusts how aggressively SQLite writes to disk (performance vs safety)     |
| `PRAGMA cache_size = 10000;`     | Sets the number of pages to hold in memory cache                           |
| `PRAGMA user_version = 1;`       | Sets a user-defined version number for application-level schema control    |
| `PRAGMA integrity_check;`        | Runs a database consistency check                                           |
| `PRAGMA temp_store = MEMORY;`    | Stores temporary tables in memory instead of disk                          |

---

### Example Usage

```sql
-- Check table schema
PRAGMA table_info('employees');

-- Enable foreign keys
PRAGMA foreign_keys = ON;

-- Change journal mode
PRAGMA journal_mode = WAL;
```

---

### Notes

- PRAGMAs are not portable across database systems (they’re specific to the database engine).
- Other databases like **Oracle, PostgreSQL, MySQL** don’t use `PRAGMA`; instead, they use configuration parameters or SQL functions/views for similar purposes.

Would you like a comparison table of configuration approaches across popular databases?