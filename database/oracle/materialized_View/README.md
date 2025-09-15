

In **Oracle**, a **Materialized View (MV)** is a database object that stores the result of a query physically, like a snapshot.
You can configure it to **auto-refresh** (periodically update) so it stays in sync with the base tables.

---

## 🔹 Creating a Materialized View with Auto Refresh

### 1. Basic Syntax

```sql
CREATE MATERIALIZED VIEW mv_name
BUILD IMMEDIATE
REFRESH [FAST | COMPLETE | FORCE]
ON [COMMIT | DEMAND | START WITH (time) NEXT (interval)]
AS
SELECT ...
FROM base_table;
```

---

### 2. Auto Refresh Options

* **FAST**: Refreshes only changed rows (needs **materialized view logs** on base tables).

* **COMPLETE**: Recomputes the entire query (slower but no logs needed).

* **FORCE**: Tries FAST, falls back to COMPLETE if not possible.

* **ON COMMIT**: Refreshes automatically whenever a transaction commits (only works with **FAST refresh**).

* **ON DEMAND**: Refreshes only when manually executed (`DBMS_MVIEW.REFRESH`).

* **START WITH / NEXT**: Sets up a **scheduled refresh**.

---

### 3. Example: Scheduled Refresh Every 5 Minutes

```sql
CREATE MATERIALIZED VIEW sales_summary_mv
BUILD IMMEDIATE
REFRESH FAST
START WITH SYSDATE
NEXT SYSDATE + (5/1440)   -- 5 minutes (1 day = 1440 minutes)
AS
SELECT product_id, SUM(amount) AS total_sales
FROM sales
GROUP BY product_id;
```

✅ This will auto-refresh every 5 minutes.

---

### 4. Example: Auto Refresh on Commit

First, create a **materialized view log** on the base table:

```sql
CREATE MATERIALIZED VIEW LOG ON sales
WITH ROWID, SEQUENCE (product_id, amount)
INCLUDING NEW VALUES;
```

Then create the MV:

```sql
CREATE MATERIALIZED VIEW sales_mv
BUILD IMMEDIATE
REFRESH FAST ON COMMIT
AS
SELECT product_id, SUM(amount) AS total_sales
FROM sales
GROUP BY product_id;
```

✅ Every time new data is committed into `sales`, the MV refreshes.

---

### 5. Manual Refresh (On Demand)

```sql
EXEC DBMS_MVIEW.REFRESH('SALES_SUMMARY_MV', 'C'); -- C = Complete
EXEC DBMS_MVIEW.REFRESH('SALES_SUMMARY_MV', 'F'); -- F = Fast
```

---

## 🔹 Key Points

* Use **FAST + ON COMMIT** for near real-time, but needs **logs**.
* Use **COMPLETE + NEXT interval** for heavy aggregations, scheduled refresh.
* Large MVs benefit from **partitioning** to speed up refresh.

---

👉 Would you like me to also explain **how Oracle decides between FAST vs COMPLETE refresh internally** (and why sometimes a "FAST refresh" silently falls back to COMPLETE)?
