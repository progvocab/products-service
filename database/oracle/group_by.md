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


# 2. **Find the Highest Salary for Each Department**

```sql
SELECT department_name, MAX(salary) AS highest_salary
FROM employees
GROUP BY department_name
ORDER BY department_name;
```

- **`MAX(salary)`** groups the highest salary **per department**.

---

# 3. **Grand Total Example**

Suppose you want **department-wise total salaries** + **grand total** at the bottom.

```sql
SELECT 
    department_name, 
    SUM(salary) AS total_salary
FROM employees
GROUP BY ROLLUP (department_name);
```

- `ROLLUP(department_name)` will automatically add one extra row:
  - For each department: department_name + sum
  - Last row: `NULL` department_name + grand total salary.

---

# **Sample Data**

| emp_id | emp_name | department_name | salary |
|:------:|:--------:|:---------------:|:------:|
| 1      | Bob      | IT               | 5000   |
| 2      | Alice    | IT               | 6000   |
| 3      | Carol    | HR               | 4000   |
| 4      | Dave     | SALES            | 5500   |
| 5      | Eve      | SALES            | 6500   |

---

# **Query Outputs**

**Second Highest Salary**
- Salary: `6000`

**Highest Salary for Each Department**

| department_name | highest_salary |
|:---------------:|:--------------:|
| HR              | 4000           |
| IT              | 6000           |
| SALES           | 6500           |

**Grand Total of Salaries**

| department_name | total_salary |
|:---------------:|:------------:|
| HR              | 4000         |
| IT              | 11000        |
| SALES           | 12000        |
| (NULL)          | 27000        |

- NULL department name means **grand total**.

---

# **Bonus**
If you want to find **second highest salary per department**, let me know — I can show that too (a slightly more advanced query).

---

Would you also want:
- **Spring Data JPA custom @Query** examples for these?
- **JdbcTemplate** version?
- **Optimized versions** for **very large data** (using analytic functions like `DENSE_RANK()`)?

I can quickly add!