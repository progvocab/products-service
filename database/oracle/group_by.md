Sure!  
Here’s how you can **find the count of employees in each department** in **Oracle SQL**:

---

# **Basic Query**

Suppose you have a table:

```text
Table: employees
Columns: emp_id, emp_name, department_name, salary, etc.
```

The query to **find the count of employees department-wise** would be:

```sql
SELECT department_name, COUNT(*) AS employee_count
FROM employees
GROUP BY department_name
ORDER BY department_name;
```

### Example output:

| department_name | employee_count |
|:----------------|:---------------|
| HR              | 10             |
| IT              | 25             |
| SALES           | 15             |

---

# **If You Have department_id instead of department_name**

Suppose employees have only **department_id** and you need to join to **departments** table.

```sql
SELECT d.department_name, COUNT(e.emp_id) AS employee_count
FROM employees e
JOIN departments d ON e.department_id = d.department_id
GROUP BY d.department_name
ORDER BY d.department_name;
```

---

# **If You Want to Include Departments with Zero Employees**

Sometimes, departments may not have any employees yet.  
You can use **LEFT JOIN** to include them:

```sql
SELECT d.department_name, COUNT(e.emp_id) AS employee_count
FROM departments d
LEFT JOIN employees e ON e.department_id = d.department_id
GROUP BY d.department_name
ORDER BY d.department_name;
```
- Departments without employees will show count = 0.

---

# **Advanced — Use ROLLUP for Subtotals**

If you want **count per department + grand total**:

```sql
SELECT 
    department_name, 
    COUNT(*) AS employee_count
FROM employees
GROUP BY ROLLUP (department_name);
```

The last row will show **total number of employees** across all departments.

---

# **Design pattern point**

- **SQL GROUP BY** here implements **Aggregation pattern** (from database design patterns).

---

Would you also like me to show:

- How to do the same using **Spring Data JPA** (`@Query`)  
- How to do the same using **Spring JdbcTemplate**  
- How to optimize it if there are **millions of rows** using **partition-wise aggregation**  

Let me know!  
I can prepare examples too.