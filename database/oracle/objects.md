In Oracle Database, **objects** and **types** are powerful features of the **Object-Relational Model**, allowing you to define complex data structures and store them in tables much like you would in object-oriented programming.

---

### **1. What Are Objects in Oracle?**

An **object** in Oracle is an instance of a **user-defined object type**. Think of it like an object in Java or C++ — it has **attributes (data)** and **methods (behavior)**.

#### **Example: Object Type Declaration**

```sql
CREATE OR REPLACE TYPE employee_type AS OBJECT (
  emp_id     NUMBER,
  emp_name   VARCHAR2(100),
  emp_salary NUMBER,
  MEMBER FUNCTION annual_salary RETURN NUMBER
);
```

#### **Object Type Body (method implementation)**

```sql
CREATE OR REPLACE TYPE BODY employee_type AS
  MEMBER FUNCTION annual_salary RETURN NUMBER IS
  BEGIN
    RETURN emp_salary * 12;
  END;
END;
```

---

### **2. Using Object Types in Tables**

Once a type is defined, you can use it as a column in a table:

```sql
CREATE TABLE employee_table (
  emp_info employee_type
);
```

Or you can use it in **PL/SQL variables**:

```sql
DECLARE
  emp employee_type;
BEGIN
  emp := employee_type(1, 'Alice', 5000);
  DBMS_OUTPUT.PUT_LINE(emp.annual_salary);
END;
```

---

### **3. What Are Types in Oracle?**

Oracle has **built-in types** (e.g., `NUMBER`, `VARCHAR2`) and **user-defined types**, which include:

| **Type**         | **Description** |
|------------------|-----------------|
| **OBJECT types** | Structured records with fields and methods. |
| **VARRAY types** | Arrays of a fixed max size. |
| **Nested tables**| Like SQL tables stored inside rows. |
| **REF types**    | References (like pointers) to objects in tables. |

#### **Example: VARRAY**

```sql
CREATE TYPE num_array AS VARRAY(5) OF NUMBER;
```

#### **Example: Nested Table**

```sql
CREATE TYPE string_table AS TABLE OF VARCHAR2(50);
```

---

### **4. Use Cases**

- **Object types**: Represent real-world entities like `Customer`, `Address`.
- **Nested tables/VARRAYS**: Store lists inside rows.
- **REFs**: Link rows like pointers (rarely used).

---

### **5. Key Differences Between Object & Relational Models**

| Feature         | Relational Table | Object Type |
|----------------|------------------|-------------|
| Data Structure | Flat (columns)    | Hierarchical/nested |
| Methods        | Not supported     | Supported (via member functions) |
| Reusability    | Limited           | High |
| Storage        | Tables            | Tables or variables |

---

Let me know if you’d like real-world schema examples using Oracle Objects and Types!