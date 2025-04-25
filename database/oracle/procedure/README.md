
- [Performance Improvement](performance.md)
- [Implicit and Explicit Attributes](attributes.md)
- [Use Case](usecase.md)

In Oracle, a **procedure** is a stored program written in PL/SQL (Procedural Language/Structured Query Language) that performs a specific task. Procedures are similar to functions but differ in that they **do not return a value directly** (they may return values through OUT parameters).

---

### **Key Features of Procedures in Oracle**

| Feature                      | Description |
|-----------------------------|-------------|
| **Modularity**              | Procedures help modularize complex logic into reusable blocks. |
| **Encapsulation**           | Business logic is encapsulated within the procedure, improving abstraction. |
| **Parameters**              | Supports `IN`, `OUT`, and `IN OUT` parameters to control data flow. |
| **Stored in Database**      | Stored procedures reside in the database schema and can be invoked by name. |
| **Performance**             | Precompiled and stored, improving performance during repeated execution. |
| **Security**                | Access can be restricted using `GRANT` permissions. |
| **Exception Handling**      | Can include `BEGIN...EXCEPTION...END` blocks to handle runtime errors. |
| **Dependency Management**   | Procedures are schema-level objects and managed by Oracle's dependency tracking system. |
| **Debugging**               | Oracle supports debugging stored procedures using tools like SQL Developer. |
| **Calling from Applications** | Procedures can be invoked from PL/SQL, Java, Python, or web apps using JDBC/ODBC. |

---

### **Procedure Parameter Modes**

| Mode     | Description                            |
|----------|----------------------------------------|
| `IN`     | Accepts a value from the caller (read-only). |
| `OUT`    | Sends a value back to the caller.      |
| `IN OUT` | Accepts a value and sends a (possibly updated) value back. |

---

### **Basic Syntax of a Procedure**
```sql
CREATE OR REPLACE PROCEDURE greet_user (
    p_name IN VARCHAR2
)
AS
BEGIN
    DBMS_OUTPUT.PUT_LINE('Hello, ' || p_name);
END;
```

---

### **Example with IN, OUT Parameters**
```sql
CREATE OR REPLACE PROCEDURE add_numbers (
    num1 IN NUMBER,
    num2 IN NUMBER,
    result OUT NUMBER
)
AS
BEGIN
    result := num1 + num2;
END;
```

**Call the Procedure:**
```sql
DECLARE
    res NUMBER;
BEGIN
    add_numbers(10, 20, res);
    DBMS_OUTPUT.PUT_LINE('Sum is: ' || res);
END;
```

---

### **Procedure with Exception Handling**
```sql
CREATE OR REPLACE PROCEDURE divide_numbers (
    a IN NUMBER,
    b IN NUMBER,
    result OUT NUMBER
)
AS
BEGIN
    IF b = 0 THEN
        RAISE_APPLICATION_ERROR(-20001, 'Division by zero');
    END IF;
    result := a / b;
EXCEPTION
    WHEN OTHERS THEN
        DBMS_OUTPUT.PUT_LINE('Error: ' || SQLERRM);
END;
```

---

### **Useful Views for Procedure Management**
- `USER_PROCEDURES`
- `ALL_PROCEDURES`
- `DBA_PROCEDURES`
- `USER_SOURCE` (to view the procedure body)

---

Let me know if you’d like examples on:
- Recursive procedures
- Procedures inside packages
- Executing from external clients like Python/Java
