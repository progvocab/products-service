Here’s how you can **list, update, and delete views** in SQLite:

---

## **1. List Views**
To list all the views in your SQLite database:

```sql
SELECT name FROM sqlite_master WHERE type = 'view';
```
This query retrieves the names of all views stored in the database.

Alternatively, if you're using the SQLite command line, you can use:
```sh
.tables
```
This will show all tables and views.

---

## **2. Update a View**
SQLite **does not support direct view modification** using `ALTER VIEW`. Instead, you must **drop and recreate** the view.

### **Step 1: Drop the Existing View**
```sql
DROP VIEW IF EXISTS user_view;
```

### **Step 2: Recreate the View with Modifications**
```sql
CREATE VIEW user_view AS 
SELECT id, name, email, age FROM users WHERE age > 21;
```

This updates the existing view with new logic.

---

## **3. Delete a View**
To remove a view from the database:

```sql
DROP VIEW IF EXISTS user_view;
```
This ensures the view is deleted without causing an error if it doesn’t exist.

---

### **Example: Managing Views**
Let's say we have this initial view:
```sql
CREATE VIEW user_view AS 
SELECT id, name, email FROM users WHERE age > 18;
```
Now, we want to **update** it to include `age` and change the age condition to `> 21`.

1. **Drop the view**:
   ```sql
   DROP VIEW IF EXISTS user_view;
   ```
2. **Recreate the view**:
   ```sql
   CREATE VIEW user_view AS 
   SELECT id, name, email, age FROM users WHERE age > 21;
   ```

---

### **Summary**
| **Action**  | **SQLite Command** |
|------------|------------------|
| List Views | `SELECT name FROM sqlite_master WHERE type = 'view';` |
| Update a View | `DROP VIEW IF EXISTS view_name;` then `CREATE VIEW view_name AS ...` |
| Delete a View | `DROP VIEW IF EXISTS view_name;` |

Would you like more examples or clarification? 🚀
