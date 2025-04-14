In Oracle PL/SQL, when working with **cursors**, attributes can be **implicit** or **explicit**, depending on how the cursor is defined and used.

Let’s break both down with examples 👇

---

## 🔸 **Implicit Cursor Attributes**

Oracle automatically creates an **implicit cursor** whenever you execute a **DML statement** (`INSERT`, `UPDATE`, `DELETE`, or `SELECT INTO`) without explicitly defining a cursor.

### ✅ Common Implicit Cursor Attributes

| Attribute           | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| `SQL%FOUND`         | Returns `TRUE` if at least one row was affected.                           |
| `SQL%NOTFOUND`      | Returns `TRUE` if no rows were affected.                                   |
| `SQL%ROWCOUNT`      | Returns the number of rows affected by the DML statement.                  |
| `SQL%ISOPEN`        | Always returns `FALSE` for implicit cursors (they close automatically).     |

### 🔍 Example

```plsql
BEGIN
    DELETE FROM employees WHERE department_id = 10;

    IF SQL%FOUND THEN
        DBMS_OUTPUT.PUT_LINE(SQL%ROWCOUNT || ' rows deleted.');
    ELSE
        DBMS_OUTPUT.PUT_LINE('No rows deleted.');
    END IF;
END;
```

---

## 🔹 **Explicit Cursor Attributes**

When you **explicitly declare a cursor**, Oracle gives you similar attributes, but they are **prefixed by the cursor name**.

### ✅ Common Explicit Cursor Attributes

| Attribute           | Description                                                           |
|---------------------|-----------------------------------------------------------------------|
| `cursor_name%FOUND`     | Returns `TRUE` if the last fetch returned a row.                      |
| `cursor_name%NOTFOUND`  | Returns `TRUE` if the last fetch did not return a row.                |
| `cursor_name%ROWCOUNT`  | Number of rows fetched so far.                                       |
| `cursor_name%ISOPEN`    | Returns `TRUE` if the cursor is currently open.                      |

### 🔍 Example

```plsql
DECLARE
    CURSOR emp_cur IS SELECT employee_id, last_name FROM employees;
    v_id employees.employee_id%TYPE;
    v_name employees.last_name%TYPE;
BEGIN
    OPEN emp_cur;
    
    LOOP
        FETCH emp_cur INTO v_id, v_name;
        EXIT WHEN emp_cur%NOTFOUND;
        
        DBMS_OUTPUT.PUT_LINE('Employee ID: ' || v_id || ', Name: ' || v_name);
    END LOOP;

    CLOSE emp_cur;
    
    DBMS_OUTPUT.PUT_LINE('Total rows processed: ' || emp_cur%ROWCOUNT);
END;
```

---

### ✅ Summary Table

| Attribute Type | Attribute         | Scope                      | Use Case Example        |
|----------------|-------------------|----------------------------|--------------------------|
| Implicit       | `SQL%FOUND`       | After DML or SELECT INTO   | `IF SQL%FOUND THEN...`   |
| Implicit       | `SQL%ROWCOUNT`    | After DML                  | `SQL%ROWCOUNT`           |
| Explicit       | `c%ISOPEN`        | During cursor operations   | `IF c%ISOPEN THEN...`    |
| Explicit       | `c%NOTFOUND`      | After `FETCH`              | `EXIT WHEN c%NOTFOUND`   |

---

Let me know if you'd like a demo of cursor FOR loops or parameterized cursors too!
