Of course! Let's go step-by-step:

---

# 1. **Find the Second Highest Salary**

Simple way using `ROWNUM` or `ROW_NUMBER()` in **Oracle SQL**:

**Approach 1: Using `ROW_NUMBER()`**

```sql
SELECT salary
FROM (
    SELECT salary, ROW_NUMBER() OVER (ORDER BY salary DESC) AS rnk
    FROM employees
) 
WHERE rnk = 2;
```

- First, order salaries descending.
- Assign row numbers.
- Pick the one with `rnk = 2` → second highest salary.

---

