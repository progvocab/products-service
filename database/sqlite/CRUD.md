In SQLite, CRUD operations (Create, Read, Update, Delete) are performed using SQL commands. Here are the basic commands:

### **1. Create (INSERT)**
To insert new records into a table:

```sql
INSERT INTO table_name (column1, column2, column3) 
VALUES ('value1', 'value2', 'value3');
```

Example:
```sql
INSERT INTO users (name, email, age) 
VALUES ('John Doe', 'john@example.com', 30);
```

---

### **2. Read (SELECT)**
To retrieve data from a table:

```sql
SELECT column1, column2 FROM table_name WHERE condition;
```

Example:
```sql
SELECT name, email FROM users WHERE age > 25;
```

To retrieve all columns:
```sql
SELECT * FROM users;
```

---

### **3. Update (UPDATE)**
To modify existing records:

```sql
UPDATE table_name 
SET column1 = 'new_value' 
WHERE condition;
```

Example:
```sql
UPDATE users 
SET email = 'newemail@example.com' 
WHERE name = 'John Doe';
```

---

### **4. Delete (DELETE)**
To remove records from a table:

```sql
DELETE FROM table_name WHERE condition;
```

Example:
```sql
DELETE FROM users WHERE age < 18;
```

To delete all records but keep the table structure:
```sql
DELETE FROM users;
```

---

### **Additional: Table Creation**
Before performing CRUD operations, you need a table:

```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    age INTEGER
);
```

Would you like help with a specific query or a more advanced example? 🚀
