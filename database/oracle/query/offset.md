In Oracle SQL, OFFSET is a clause used with FETCH to skip a specified number of rows in the result set before starting to return rows. It’s commonly used for pagination (e.g., page 1, page 2, etc.).

✅ Basic Syntax

SELECT columns
FROM table_name
ORDER BY some_column
OFFSET n ROWS
FETCH NEXT m ROWS ONLY;

OFFSET n ROWS → skip n rows

FETCH NEXT m ROWS ONLY → return only m rows after the offset


✔️ Example: Pagination

Get rows 11–20 from a table:

SELECT *
FROM employees
ORDER BY employee_id
OFFSET 10 ROWS
FETCH NEXT 10 ROWS ONLY;

⚠️ Important Notes

OFFSET must be used with an ORDER BY clause; otherwise, results may be unpredictable.

Oracle added OFFSET/FETCH in Oracle 12c — older versions use ROWNUM instead.


✔️ Equivalent Using ROWNUM (Old Method)

SELECT *
FROM (
   SELECT e.*, ROW_NUMBER() OVER (ORDER BY employee_id) AS rn
   FROM employees e
)
WHERE rn > 10 AND rn <= 20;


---

If you want, I can show you how to implement pagination in Oracle, or rewrite your query using OFFSET.