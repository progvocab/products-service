In **Oracle**, an **editionable function** is a **PL/SQL function that can be associated with a specific database edition**, allowing you to have multiple **versions** of the same function in different editions without affecting others.

### Key Points:

* **Editionable** objects support **online application upgrades**.
* You can modify the function in a new **edition** while the old edition continues to serve existing sessions.
* Typical usage: **zero-downtime patching or schema evolution**.
* By default, PL/SQL functions are **editionable** unless declared `NONEDITIONABLE`.

### Example:

```sql
CREATE OR REPLACE FUNCTION calculate_bonus(salary NUMBER)
RETURN NUMBER
EDITIONABLE
IS
BEGIN
    RETURN salary * 0.1;
END;
/
```

💡 **In short:**

> An editionable function allows you to maintain multiple versions of the function across **database editions**, enabling safe, online application upgrades.
