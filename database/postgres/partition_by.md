### **`PARTITION BY` in ANSI SQL**

The `PARTITION BY` clause is used with **window functions** to divide the result set into partitions (groups of rows) before applying a **window function** like `ROW_NUMBER()`, `RANK()`, `SUM()`, etc.

**Syntax Example (ANSI SQL compliant):**

```sql
SELECT
  employee_id,
  department_id,
  salary,
  RANK() OVER (PARTITION BY department_id ORDER BY salary DESC) AS dept_rank
FROM employees;
```

- This ranks employees **within each department** by salary.
- `PARTITION BY department_id` creates a partition for each department.
- `ORDER BY salary DESC` defines how rows in each partition are sorted.

---

### **In PostgreSQL**

PostgreSQL **fully supports** `PARTITION BY` as defined in ANSI SQL.

**PostgreSQL Example:**

```sql
SELECT
  employee_id,
  department_id,
  salary,
  DENSE_RANK() OVER (PARTITION BY department_id ORDER BY salary DESC) AS rank
FROM employees;
```

- PostgreSQL also supports advanced window frame options like `ROWS BETWEEN`, `RANGE`, etc.

---

### **In Oracle**

Oracle also fully supports `PARTITION BY` in both:
- **Analytic functions (Window Functions)** — similar to ANSI SQL.
- **Partitioned Tables** — which is different and refers to physical data partitioning.

**Oracle Analytic Function Example:**

```sql
SELECT
  employee_id,
  department_id,
  salary,
  ROW_NUMBER() OVER (PARTITION BY department_id ORDER BY salary DESC) AS row_num
FROM employees;
```

**Note:** Oracle also allows usage of `PARTITION BY` in `MODEL` and `MATCH_RECOGNIZE` clauses.

---

### **Comparison Table**

| Feature                        | ANSI SQL           | PostgreSQL         | Oracle             |
|-------------------------------|--------------------|--------------------|--------------------|
| Basic Window Partitioning     | Yes                | Yes                | Yes                |
| Use in Analytic Functions     | Yes                | Yes                | Yes                |
| Partitioned Tables            | No (not standard)  | Yes (native support) | Yes (extensive)   |
| Frame Specification (`ROWS BETWEEN`) | Yes         | Yes                | Yes                |
| Custom Aggregates in Partitions| Limited            | Yes (custom window functions) | Yes (analytic functions) |

---

Would you like me to show examples involving `SUM()` with `PARTITION BY` for running totals or compare physical vs logical partitioning?