Here’s a collection of **important PostgreSQL commands** for managing **databases, tables, schemas, and users**, along with examples.

---

## **1. Database Management**
### **View Existing Databases**
```sql
\l      -- List all databases
\du     -- List all users
```
or
```sql
SELECT datname FROM pg_database;
```

### **Create a Database**
```sql
CREATE DATABASE my_database;
```

### **Connect to a Database**
```sql
\c my_database;
```

### **Delete (Drop) a Database**
```sql
DROP DATABASE my_database;
```
⚠ **Note:** Cannot be undone!

---

## **2. Table Management**
### **View Existing Tables**
```sql
\dt      -- List all tables in the current database
\dt schema_name.*  -- List tables in a specific schema
```
or
```sql
SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';
```

### **Create a Table**
```sql
CREATE TABLE employees (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    department VARCHAR(50),
    salary DECIMAL(10,2),
    joined_on DATE DEFAULT CURRENT_DATE
);
```

### **View Table Structure**
```sql
\d employees
```
or
```sql
SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'employees';
```

### **Rename a Table**
```sql
ALTER TABLE employees RENAME TO staff;
```

### **Delete (Drop) a Table**
```sql
DROP TABLE employees;
```

---

## **3. Schema Management**
### **View Existing Schemas**
```sql
SELECT schema_name FROM information_schema.schemata;
```
or
```sql
\dn
```

### **Create a Schema**
```sql
CREATE SCHEMA my_schema;
```

### **Set Default Schema for a User**
```sql
ALTER ROLE my_user SET search_path TO my_schema;
```

### **Delete (Drop) a Schema**
```sql
DROP SCHEMA my_schema CASCADE;
```
⚠ `CASCADE` will remove all objects inside the schema!

---

## **4. User Management**
### **View Existing Users**
```sql
\du
```

### **Create a User**
```sql
CREATE USER my_user WITH PASSWORD 'mypassword';
```

### **Grant Privileges to a User**
```sql
GRANT ALL PRIVILEGES ON DATABASE my_database TO my_user;
```
or for specific tables:
```sql
GRANT SELECT, INSERT, UPDATE ON employees TO my_user;
```

### **Change User Password**
```sql
ALTER USER my_user WITH PASSWORD 'newpassword';
```

### **Delete (Drop) a User**
```sql
DROP USER my_user;
```

---

## **5. Additional Useful Commands**
### **View Active Connections**
```sql
SELECT * FROM pg_stat_activity;
```

### **Terminate a Connection**
```sql
SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = 'my_database';
```

### **Backup a Database**
```bash
pg_dump -U postgres -W -F c -b -v -f "backup.dump" my_database
```

### **Restore a Database**
```bash
pg_restore -U postgres -d my_database -v "backup.dump"
```

---

### **Summary Table**
| **Operation** | **Command** |
|--------------|------------|
| View Databases | `\l` |
| Create Database | `CREATE DATABASE db_name;` |
| Drop Database | `DROP DATABASE db_name;` |
| View Tables | `\dt` |
| Create Table | `CREATE TABLE table_name (...);` |
| Drop Table | `DROP TABLE table_name;` |
| View Schemas | `\dn` |
| Create Schema | `CREATE SCHEMA schema_name;` |
| Drop Schema | `DROP SCHEMA schema_name CASCADE;` |
| View Users | `\du` |
| Create User | `CREATE USER username WITH PASSWORD 'password';` |
| Grant Privileges | `GRANT ALL PRIVILEGES ON DATABASE db_name TO username;` |
| Drop User | `DROP USER username;` |

Would you like more details on any specific command?