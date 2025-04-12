In Oracle, a **Database Link (DB Link)** is a schema object that allows you to connect to another database and query or perform operations on its data **as if it were local**.

---

### **1. Purpose of DB Link**

It enables **cross-database communication**, especially useful when:

- Consolidating data from multiple databases
- Migrating or syncing data between environments
- Creating distributed applications

---

### **2. Types of DB Links**

| Type               | Description |
|--------------------|-------------|
| **Private**        | Visible only to the user who created it |
| **Public**         | Visible to all users in the database |
| **Fixed user**     | Always connects using a specified user |
| **Connected user** | Uses the credentials of the current user |
| **Current user**   | Uses the proxy authentication mechanism |

---

### **3. Creating a DB Link**

#### **Syntax (Connected User)**

```sql
CREATE DATABASE LINK hr_link
CONNECT TO hr IDENTIFIED BY hr_password
USING 'remote_tns';
```

- `hr` and `hr_password` are credentials on the **remote database**.
- `'remote_tns'` refers to a **TNS alias** in `tnsnames.ora`.

#### **Public Link Example**

```sql
CREATE PUBLIC DATABASE LINK sales_link
CONNECT TO sales_user IDENTIFIED BY sales_pass
USING 'salesdb';
```

---

### **4. Using a DB Link**

You can query remote data like this:

```sql
SELECT * FROM employees@hr_link;
```

You can also insert data remotely:

```sql
INSERT INTO orders@remote_link (id, product) VALUES (1, 'Laptop');
```

Or join local and remote tables:

```sql
SELECT a.name, b.salary
FROM local_employees a
JOIN employees@hr_link b ON a.id = b.id;
```

---

### **5. Checking Existing DB Links**

```sql
SELECT * FROM USER_DB_LINKS;
-- or
SELECT * FROM ALL_DB_LINKS;
-- or
SELECT * FROM DBA_DB_LINKS;
```

---

### **6. Security Considerations**

- Avoid using **public DB links** unless necessary.
- Ensure **encrypted connections** with `Oracle Net Encryption`.
- Prefer **using wallet-based authentication** over hardcoded passwords.

---

Let me know if you’d like help setting up a secure DB link with `sqlnet.ora` and `wallet` authentication!