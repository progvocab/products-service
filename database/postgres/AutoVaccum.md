# AutoVacuum

In PostgreSQL, **autovacuum is triggered per table** based on dead tuples, not time.

1. **VACUUM trigger**:
   `dead_tuples > autovacuum_vacuum_threshold + autovacuum_vacuum_scale_factor × reltuples`

2. **Default values**:
   `autovacuum_vacuum_threshold = 50`
   `autovacuum_vacuum_scale_factor = 0.2`

3. **ANALYZE trigger**:
   `modified_tuples > autovacuum_analyze_threshold + autovacuum_analyze_scale_factor × reltuples`

4. **Default ANALYZE values**:
   `autovacuum_analyze_threshold = 50`, `autovacuum_analyze_scale_factor = 0.1`

5. Settings can be **overridden per table** using `ALTER TABLE … SET (autovacuum_*)`.


### Write Operation

1. Adding a new row version usually causes a **write to a random heap page**, not sequential I/O.
2. PostgreSQL must find a page with **enough free space**, which can be anywhere in the table.
3. The **Free Space Map (FSM)** guides this choice, leading to non-sequential writes.
4. Only pure bulk inserts into an empty or append-only table tend toward sequential I/O.
5. Updates therefore generate **random I/O and more WAL** than inserts.
1. **Dead tuples** are old row versions that are no longer visible to any active transaction.
2. They are created by **UPDATE** (old row version) and **DELETE** operations.
3. PostgreSQL keeps them due to **MVCC**, instead of overwriting rows in place.
4. Dead tuples **consume disk space** and slow down table and index scans.
5. They are removed and space is reclaimed by **VACUUM / autovacuum**.
7. Dead tuples are stored **in the same heap table pages** as live rows.
2. PostgreSQL never removes rows immediately due to **MVCC visibility rules**.
3. The tuple header is marked as **dead/invalid** but the data remains on disk.
4. Index entries may still point to these dead tuples until vacuum runs.
5. **VACUUM** marks the space reusable inside the same table pages.

