### **Window Functions in PostgreSQL**
Window functions in PostgreSQL allow performing calculations across a set of table rows related to the current row. Unlike aggregate functions, window functions do not collapse rows; instead, they return a value for each row while considering a specific "window" (a subset of rows).

---

## **1. Syntax of a Window Function**
```sql
function_name (expression) 
OVER (
    PARTITION BY column_name 
    ORDER BY column_name 
    ROWS BETWEEN start AND end
)
```
- **`PARTITION BY`**: Divides the dataset into partitions (like `GROUP BY` but keeps all rows).
- **`ORDER BY`**: Specifies the order in which the function processes rows.
- **`ROWS BETWEEN`**: Defines the frame within which the calculation is performed.

---

## **2. Types of Window Functions**
### **A. Ranking Functions**
#### 1️⃣ `ROW_NUMBER()` - Assigns a unique row number
```sql
SELECT id, department, salary,
       ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) AS row_num
FROM employees;
```
- Resets row numbers for each department.

#### 2️⃣ `RANK()` - Assigns rank with gaps
```sql
SELECT id, department, salary,
       RANK() OVER (PARTITION BY department ORDER BY salary DESC) AS rank
FROM employees;
```
- If two employees have the same salary, they get the same rank, but the next rank is skipped.

#### 3️⃣ `DENSE_RANK()` - Assigns rank without gaps
```sql
SELECT id, department, salary,
       DENSE_RANK() OVER (PARTITION BY department ORDER BY salary DESC) AS dense_rank
FROM employees;
```
- Similar to `RANK()` but without skipping numbers.

#### 4️⃣ `NTILE(n)` - Divides rows into `n` groups
```sql
SELECT id, salary,
       NTILE(4) OVER (ORDER BY salary) AS quartile
FROM employees;
```
- Distributes salaries into 4 quartiles.

---

### **B. Aggregate Window Functions**
Unlike normal aggregates, these keep all rows.

#### 5️⃣ `SUM()` - Running total
```sql
SELECT id, department, salary,
       SUM(salary) OVER (PARTITION BY department ORDER BY id) AS running_total
FROM employees;
```
- Calculates a cumulative sum within each department.

#### 6️⃣ `AVG()` - Running average
```sql
SELECT id, department, salary,
       AVG(salary) OVER (PARTITION BY department ORDER BY id) AS running_avg
FROM employees;
```
- Computes moving average of salaries.

#### 7️⃣ `MIN() / MAX()` - Rolling min/max
```sql
SELECT id, department, salary,
       MIN(salary) OVER (PARTITION BY department ORDER BY id) AS rolling_min,
       MAX(salary) OVER (PARTITION BY department ORDER BY id) AS rolling_max
FROM employees;
```

---

### **C. Value-Based Window Functions**
#### 8️⃣ `LAG()` - Previous row’s value
```sql
SELECT id, department, salary,
       LAG(salary, 1, 0) OVER (PARTITION BY department ORDER BY id) AS prev_salary
FROM employees;
```
- Retrieves the previous salary (defaults to `0` if no previous row exists).

#### 9️⃣ `LEAD()` - Next row’s value
```sql
SELECT id, department, salary,
       LEAD(salary, 1, 0) OVER (PARTITION BY department ORDER BY id) AS next_salary
FROM employees;
```
- Fetches the next row’s salary.

#### 🔟 `FIRST_VALUE()` - First row's value in partition
```sql
SELECT id, department, salary,
       FIRST_VALUE(salary) OVER (PARTITION BY department ORDER BY id) AS min_salary
FROM employees;
```

#### 1️⃣1️⃣ `LAST_VALUE()` - Last row's value in partition
```sql
SELECT id, department, salary,
       LAST_VALUE(salary) OVER (PARTITION BY department ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS max_salary
FROM employees;
```
- Without defining a frame, it may return unexpected results.

---

## **3. Window Frame Definitions**
You can define how many rows a function considers:
```sql
ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
```
- Example: Cumulative sum
```sql
SELECT id, salary,
       SUM(salary) OVER (ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_total
FROM employees;
```

---

## **4. Use Cases**
- Running totals (e.g., cumulative sales)
- Rank-based reporting (e.g., leaderboard)
- Comparing current and previous rows (e.g., stock price changes)
- Moving averages (e.g., sales trends)

Would you like a specific example tailored to your use case?

### **`QUALIFY` in PostgreSQL**
In **PostgreSQL**, there is no native `QUALIFY` clause like in **BigQuery** or **Snowflake**. However, its functionality can be replicated using a **subquery** or **Common Table Expressions (CTE)**.

---

## **1. What is `QUALIFY` in Other Databases?**
In databases like BigQuery and Snowflake, `QUALIFY` is used to filter results **after** window functions are applied.  
For example, filtering only the top-ranked row per category:

```sql
SELECT id, department, salary,
       RANK() OVER (PARTITION BY department ORDER BY salary DESC) AS rank
FROM employees
QUALIFY rank = 1;
```

---

## **2. PostgreSQL Alternative to `QUALIFY`**
Since PostgreSQL does not support `QUALIFY`, we must use a **`WHERE` clause in a subquery** or **CTE**.

### **A. Using a Subquery**
```sql
SELECT * FROM (
    SELECT id, department, salary,
           RANK() OVER (PARTITION BY department ORDER BY salary DESC) AS rank
    FROM employees
) ranked
WHERE rank = 1;
```

### **B. Using a Common Table Expression (CTE)**
```sql
WITH ranked_employees AS (
    SELECT id, department, salary,
           RANK() OVER (PARTITION BY department ORDER BY salary DESC) AS rank
    FROM employees
)
SELECT * FROM ranked_employees WHERE rank = 1;
```

---

## **3. Why Use `QUALIFY` (or Its Alternative)?**
- When using window functions, `WHERE` cannot filter their results because it acts **before** window functions execute.
- Using **subqueries or CTEs** ensures filtering occurs **after** the window function.

Would you like a more complex example, like filtering top `N` employees per department?
