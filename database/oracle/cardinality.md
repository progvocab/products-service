**Cardinality** in Oracle (or any relational database) refers to the **number of distinct values** in a column.

---

## **High vs Low Cardinality**

| Type             | Description                                           | Example Column      | Use Case / Scenario |
|------------------|-------------------------------------------------------|----------------------|----------------------|
| **High Cardinality** | Many **unique/distinct** values                     | `email_id`, `user_id`, `phone_number` | Ideal for **B-tree indexes** |
| **Low Cardinality**  | Few **repeating values**                           | `gender`, `status` (`active/inactive`) | Better with **bitmap indexes** |
| **Medium Cardinality** | Moderate amount of unique values (e.g., `city`, `dept_id`) | `department`, `region` | Depends on data distribution |

---

## **Example Scenarios**

### 1. **High Cardinality Example**

```sql
SELECT COUNT(DISTINCT employee_id) FROM employees;
-- Assume employee_id is unique, this is high cardinality.
```

- Indexing on `employee_id` will be very effective (B-tree index).
- Useful for point queries:

```sql
SELECT * FROM employees WHERE employee_id = 105;
```

---

### 2. **Low Cardinality Example**

```sql
SELECT COUNT(DISTINCT gender) FROM employees;
-- Only two values: 'M', 'F'
```

- Not efficient for B-tree.
- Better to use **bitmap index**:

```sql
CREATE BITMAP INDEX idx_gender ON employees(gender);
```

- Effective in queries like:

```sql
SELECT * FROM employees WHERE gender = 'F';
```

---

### 3. **How to Check Cardinality in Oracle**

You can use `DBMS_STATS` to gather and check cardinality:

```sql
BEGIN
  DBMS_STATS.GATHER_TABLE_STATS('HR', 'EMPLOYEES');
END;
```

Then query:

```sql
SELECT column_name, num_distinct
FROM all_tab_col_statistics
WHERE table_name = 'EMPLOYEES' AND owner = 'HR';
```

---

## **When Does Cardinality Matter?**

- **Query optimization**: Oracle uses cardinality to choose the best execution plan.
- **Index selection**: Bitmap for low cardinality; B-tree for high.
- **Join performance**: Helps decide join order and type (nested loop vs hash).

---

Let me know if you want examples with **execution plans** or **explain analyze**.