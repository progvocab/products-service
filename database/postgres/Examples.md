Here are the **most common and technically correct reasons** why a PostgreSQL DB instance suddenly consumes **90 GB out of 100 GB**, even if your actual table data is much smaller.

All points are concise, to-the-point, and relevant for AWS RDS or self-managed PostgreSQL.

### **Data Bloat & Unvacuumed Dead Tuples**

PostgreSQL uses **MVCC**, meaning updates/deletes create **dead tuples**.
If `autovacuum` is too slow or misconfigured, these accumulate, causing:

* Table bloat
* Index bloat
* TOAST table growth
* Huge storage without real data increase

**Check bloat**

```sql
SELECT relname, n_dead_tup 
FROM pg_stat_all_tables 
WHERE n_dead_tup > 100000;
```

### **Index Bloat**

Indexes grow continuously with updates.
Index bloat is often **worse than table bloat**.

Check size:

```sql
SELECT relname AS index_name, pg_size_pretty(pg_relation_size(indexrelid))
FROM pg_stat_user_indexes
JOIN pg_index USING (indexrelid)
ORDER BY pg_relation_size(indexrelid) DESC;
```

### **Large WAL / Write-Ahead Logs (RDS often hit this)**

Reasons:

* High write throughput
* Long-running queries delaying checkpoint
* Replica lag (WAL retained for replica)
* Logical replication slots not consumed
* WAL archiving accumulation

**Check WAL directory usage (RDS)**:

```sql
SELECT * FROM pg_replication_slots;
```

If `restart_lsn` is stuck → WAL cannot be recycled.

### **Long-Running Transactions Holding Vacuum**

A long-running transaction prevents vacuum cleanup.

Check:

```sql
SELECT pid, age(xact_start), query 
FROM pg_stat_activity
WHERE state <> 'idle';
```

### **Autovacuum Not Running or Too Slow**

Common causes:

* `autovacuum_work_mem` too low
* `autovacuum_vacuum_cost_limit` too low
* Manual cron jobs disabled
* Table >50M rows → vacuum cannot keep up

### **Large Temporary Files**

Queries using sorts/joins/hash aggregations spill to disk.

Check:

```sql
SELECT datname, pg_size_pretty(temp_bytes)
FROM pg_stat_database;
```

### **Extensions Consuming Space**

E.g., PostGIS, TimescaleDB, pglogical.

Check per-extension objects:

```sql
SELECT * FROM pg_available_extensions;
```

### **TOAST Table Bloat**

Large JSON, TEXT, BYTEA columns create TOAST storage that grows and does not shrink automatically.

Check TOAST table size:

```sql
SELECT reltoastrelid::regclass AS toast_table,
       pg_size_pretty(pg_relation_size(reltoastrelid))
FROM pg_class
WHERE reltoastrelid <> 0;
```

### **Abandoned Replication Slots**

Most common in **RDS**.
If a replication slot is created but consumer is gone → WAL grows forever.

Check:

```sql
SELECT slot_name, active, restart_lsn FROM pg_replication_slots;
```

If any slot is inactive → it is causing space leak.

### **Manual Backups / Old Snapshots**

Self-managed Postgres only:

* Old base backups
* PITR WAL archives
* pgBackRest backups piling up

### **Large Orphaned Files in `pg_wal` or `pg_log`**

If pg_wal is stuck or logs are too large due to verbose logging.

 

## **Most Common Real Cause (Given Your Numbers)**

When a DB goes from healthy to 90 GB out of 100 GB:

### **80% probability: WAL retention due to a stuck replication slot or replica lag.**

### **15%: Autovacuum cannot keep up resulting in bloat.**

### **5%: Temp files, TOAST, extensions, etc.**
 

If you share:

* Output of `pg_stat_user_tables` (row count / dead tuples)
* Output of `pg_replication_slots`
* Output of database size summary

I can tell you the exact reason in <30 seconds.
