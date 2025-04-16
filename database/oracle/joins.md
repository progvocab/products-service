Here's a comprehensive example demonstrating **LEFT JOIN**, **RIGHT JOIN**, **INNER JOIN**, **OUTER JOIN**, and **FULL JOIN** using Oracle SQL, with `EMPLOYEE` and `INTERN` tables that have the same columns: `ID`, `NAME`, `DEPARTMENT`.

Let’s assume the following structure for both tables:

```sql
EMPLOYEE(ID, NAME, DEPARTMENT)
INTERN(ID, NAME, DEPARTMENT)
```

---

### 📘 Sample Data

| EMPLOYEE           |         | INTERN             |
|--------------------|---------|--------------------|
| ID | NAME  | DEPT   |         | ID | NAME  | DEPT   |
|----|-------|--------|         |----|-------|--------|
| 1  | John  | Sales  |         | 3  | Alice | HR     |
| 2  | Mike  | HR     |         | 4  | Bob   | Sales  |
| 3  | Alice | IT     |         | 5  | Carol | IT     |

---

### 🔸 1. INNER JOIN

Returns records that have matching values in both tables.

```sql
SELECT E.ID, E.NAME, E.DEPARTMENT, I.ID AS INTERN_ID
FROM EMPLOYEE E
INNER JOIN INTERN I
ON E.ID = I.ID;
```

**Result:** Only rows where `EMPLOYEE.ID = INTERN.ID`.

---

### 🔸 2. LEFT JOIN (or LEFT OUTER JOIN)

Returns all rows from the left table (`EMPLOYEE`), and matched rows from the right table (`INTERN`). If no match, returns NULLs.

```sql
SELECT E.ID, E.NAME, E.DEPARTMENT, I.ID AS INTERN_ID
FROM EMPLOYEE E
LEFT JOIN INTERN I
ON E.ID = I.ID;
```

**Result:** All `EMPLOYEE` records, matched with `INTERN` if available.

---

### 🔸 3. RIGHT JOIN (or RIGHT OUTER JOIN)

Returns all rows from the right table (`INTERN`), and matched rows from the left table (`EMPLOYEE`).

```sql
SELECT E.ID, E.NAME, E.DEPARTMENT, I.ID AS INTERN_ID
FROM EMPLOYEE E
RIGHT JOIN INTERN I
ON E.ID = I.ID;
```

**Result:** All `INTERN` records, matched with `EMPLOYEE` if available.

---

### 🔸 4. FULL JOIN (or FULL OUTER JOIN)

Returns all records when there is a match in either table. Where there is no match, NULLs are returned.

```sql
SELECT E.ID, E.NAME, E.DEPARTMENT, I.ID AS INTERN_ID
FROM EMPLOYEE E
FULL OUTER JOIN INTERN I
ON E.ID = I.ID;
```

**Result:** All records from both tables, with matches where available.

---

### 🔸 5. OUTER JOIN (ANSI vs Oracle Syntax)

Oracle supports **ANSI JOIN** syntax (as shown above) and **Oracle proprietary syntax** using `(+).` Here’s how a LEFT OUTER JOIN would look:

```sql
SELECT E.ID, E.NAME, E.DEPARTMENT, I.ID AS INTERN_ID
FROM EMPLOYEE E, INTERN I
WHERE E.ID = I.ID(+);
```

> This is a LEFT OUTER JOIN — `INTERN` is the optional table.

---

Let me know if you want to see results from a specific dataset or want a visualization!
