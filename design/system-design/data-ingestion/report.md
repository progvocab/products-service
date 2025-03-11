### **Finding Underpaid and Overpaid Employees in Amazon Redshift**  

To identify **underpaid** and **overpaid** employees, you need to:  
1. **Define a benchmark salary** (e.g., based on average salary, median salary, or industry standard).  
2. **Compare employee salaries** with the benchmark.  

---

## **1. Sample Employee Table**
Assume you have an **`employees`** table in Redshift:

```sql
CREATE TABLE employees (
    emp_id INT PRIMARY KEY,
    name VARCHAR(100),
    department VARCHAR(50),
    salary DECIMAL(10,2),
    job_title VARCHAR(50)
);
```

**Sample Data:**
| emp_id | name     | department  | salary  | job_title  |
|--------|---------|------------|---------|------------|
| 1      | Alice   | Engineering | 120000  | Software Engineer |
| 2      | Bob     | Engineering | 90000   | Software Engineer |
| 3      | Charlie | HR          | 60000   | HR Manager |
| 4      | Dave    | HR          | 75000   | HR Manager |
| 5      | Eve     | Engineering | 140000  | Software Engineer |

---

## **2. Find the Benchmark Salary**
You can define a benchmark using:  
- **Industry standard** (if available).  
- **Average salary per job title**.  
- **Median salary per department**.

### **Find Average Salary Per Job Title**
```sql
SELECT job_title, AVG(salary) AS avg_salary
FROM employees
GROUP BY job_title;
```
This gives an **average salary** for each job role.

---

## **3. Identify Underpaid and Overpaid Employees**
An **underpaid employee** earns **less than 80%** of the average salary.  
An **overpaid employee** earns **more than 120%** of the average salary.

### **Find Underpaid Employees**
```sql
WITH avg_salary AS (
    SELECT job_title, AVG(salary) AS avg_salary
    FROM employees
    GROUP BY job_title
)
SELECT e.emp_id, e.name, e.job_title, e.salary, a.avg_salary
FROM employees e
JOIN avg_salary a ON e.job_title = a.job_title
WHERE e.salary < a.avg_salary * 0.8;
```

### **Find Overpaid Employees**
```sql
WITH avg_salary AS (
    SELECT job_title, AVG(salary) AS avg_salary
    FROM employees
    GROUP BY job_title
)
SELECT e.emp_id, e.name, e.job_title, e.salary, a.avg_salary
FROM employees e
JOIN avg_salary a ON e.job_title = a.job_title
WHERE e.salary > a.avg_salary * 1.2;
```

---

## **4. Alternative: Find Underpaid/Overpaid Employees Based on Department**
If you want to compare **salaries within departments**, replace `job_title` with `department`:

```sql
WITH avg_salary AS (
    SELECT department, AVG(salary) AS avg_salary
    FROM employees
    GROUP BY department
)
SELECT e.emp_id, e.name, e.department, e.salary, a.avg_salary
FROM employees e
JOIN avg_salary a ON e.department = a.department
WHERE e.salary < a.avg_salary * 0.8
   OR e.salary > a.avg_salary * 1.2;
```

---

## **5. Result Example**
For a **Software Engineer**, if the average salary is **$116,666**:  
- **Bob ($90,000) is underpaid** (`90000 < 116666 * 0.8`).  
- **Eve ($140,000) is overpaid** (`140000 > 116666 * 1.2`).  

| emp_id | name | job_title         | salary  | avg_salary | Status   |
|--------|------|-------------------|---------|------------|----------|
| 2      | Bob  | Software Engineer | 90000   | 116666     | Underpaid |
| 5      | Eve  | Software Engineer | 140000  | 116666     | Overpaid |

---

## **Conclusion**
- ✅ **Redshift SQL can efficiently find underpaid and overpaid employees**.  
- ✅ **Adjust salary thresholds (80% & 120%) based on company policy**.  
- ✅ **Use `job_title` or `department` depending on analysis needs**.  

Would you like help integrating this into a Redshift-based analytics dashboard?