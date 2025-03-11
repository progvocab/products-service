### **Integrating Underpaid/Overpaid Employee Analysis into a Redshift-Based Analytics Dashboard**  

To build a **Redshift-based analytics dashboard** for identifying underpaid and overpaid employees, follow these steps:  

---

## **1. Architecture Overview**  
1. **Redshift Data Storage**: Stores employee salary data.  
2. **AWS Glue / Lambda**: Runs queries and extracts insights.  
3. **Amazon QuickSight / Streamlit / Grafana**: Visualizes salary insights.  
4. **API Layer (Spring Boot or Flask)**: Exposes insights via REST APIs.  

---

## **2. Redshift Table Setup**  
Ensure the `employees` table is structured properly:  

```sql
CREATE TABLE employees (
    emp_id INT PRIMARY KEY,
    name VARCHAR(100),
    department VARCHAR(50),
    salary DECIMAL(10,2),
    job_title VARCHAR(50)
);
```

**Sample Data Insert:**
```sql
INSERT INTO employees (emp_id, name, department, salary, job_title)
VALUES
(1, 'Alice', 'Engineering', 120000, 'Software Engineer'),
(2, 'Bob', 'Engineering', 90000, 'Software Engineer'),
(3, 'Charlie', 'HR', 60000, 'HR Manager'),
(4, 'Dave', 'HR', 75000, 'HR Manager'),
(5, 'Eve', 'Engineering', 140000, 'Software Engineer');
```

---

## **3. Create a View for Underpaid/Overpaid Employees**
A **Redshift VIEW** helps avoid redundant queries.  

```sql
CREATE OR REPLACE VIEW salary_analysis AS
WITH avg_salary AS (
    SELECT job_title, AVG(salary) AS avg_salary
    FROM employees
    GROUP BY job_title
)
SELECT e.emp_id, e.name, e.department, e.job_title, e.salary, a.avg_salary,
       CASE 
           WHEN e.salary < a.avg_salary * 0.8 THEN 'Underpaid'
           WHEN e.salary > a.avg_salary * 1.2 THEN 'Overpaid'
           ELSE 'Fairly Paid'
       END AS salary_status
FROM employees e
JOIN avg_salary a ON e.job_title = a.job_title;
```

Now, you can query:  
```sql
SELECT * FROM salary_analysis;
```

---

## **4. Expose Insights via REST API (Spring Boot)**
To integrate with a dashboard, expose the data using a REST API.

### **Spring Boot Controller (Redshift Integration)**
```java
@RestController
@RequestMapping("/salary")
public class SalaryAnalysisController {

    @Autowired
    private JdbcTemplate jdbcTemplate;

    @GetMapping("/analysis")
    public List<Map<String, Object>> getSalaryAnalysis() {
        String sql = "SELECT * FROM salary_analysis";
        return jdbcTemplate.queryForList(sql);
    }
}
```

- **Use JDBC to connect to Redshift.**
- **Call `/salary/analysis` to get underpaid/overpaid employee data.**

---

## **5. Visualizing Data in Amazon QuickSight**
### **Steps to Connect QuickSight with Redshift**
1. **Go to Amazon QuickSight** → "Manage Data".  
2. **Create a new Dataset** → Select **Redshift Connector**.  
3. **Enter Redshift Credentials & Cluster Details**.  
4. **Choose `salary_analysis` view.**  
5. **Create visualizations**:  
   - **Bar Chart**: Compare salaries per job title.  
   - **Pie Chart**: Show % of underpaid vs. overpaid employees.  

---

## **6. Alternative: Streamlit Dashboard (Python)**
If you want a **lightweight dashboard**, use **Streamlit**:

### **Install Dependencies**
```bash
pip install streamlit psycopg2
```

### **Streamlit App for Redshift Salary Analysis**
```python
import streamlit as st
import psycopg2
import pandas as pd

# Redshift Connection
conn = psycopg2.connect(
    dbname="your_db", user="your_user", password="your_pass",
    host="your_redshift_host", port="5439"
)

# Query Data
query = "SELECT * FROM salary_analysis"
df = pd.read_sql(query, conn)

# Dashboard
st.title("Salary Analysis Dashboard")
st.dataframe(df)
st.bar_chart(df.set_index("job_title")["salary"])
```

Run with:
```bash
streamlit run app.py
```

---

## **Conclusion**
- **Redshift Query Layer**: `salary_analysis` view for underpaid/overpaid employees.  
- **API Layer**: Spring Boot exposes REST API.  
- **Dashboard**: Amazon QuickSight / Streamlit for visualization.  

Would you like help with deployment on AWS?