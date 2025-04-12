Yes, **many databases developed in C++ support ANSI SQL**, but let's break this down a bit:

---

### **1. What Is ANSI SQL?**
**ANSI SQL** (or standard SQL) refers to the standardized version of SQL defined by the American National Standards Institute. It specifies:

- Syntax for queries (`SELECT`, `INSERT`, etc.)
- Data types
- Constraints
- Joins, unions
- Views, subqueries
- Transaction control (`BEGIN`, `COMMIT`, `ROLLBACK`)

---

### **2. Do C++ Databases Use ANSI SQL?**

| **Database**     | **Written In** | **Supports ANSI SQL** | **Notes** |
|------------------|----------------|------------------------|-----------|
| **MySQL**        | C/C++          | Partial                | Uses its own dialect with ANSI compliance modes. |
| **PostgreSQL**   | C              | Yes (Highly compliant) | Very close to ANSI SQL, with extensions. |
| **ClickHouse**   | C++            | Partially              | Uses its own SQL-like dialect optimized for OLAP. |
| **SQLite**       | C              | Partial                | Implements a large subset of SQL, but simplified. |
| **Oracle**       | C/C++          | Extended               | Supports ANSI SQL plus proprietary features. |
| **MariaDB**      | C/C++          | Partial                | Follows MySQL's model with enhancements. |
| **FirebirdSQL**  | C++            | Yes                    | ANSI-compliant with extensions. |

---

### **3. Why Not 100% ANSI SQL?**

- **Performance tuning:** Some engines optimize for OLAP (like ClickHouse) or OLTP, requiring non-standard syntax.
- **Features:** Window functions, JSON support, full-text search, etc., often go beyond ANSI SQL.
- **Usability:** Vendors add features like `LIMIT`/`OFFSET` (not ANSI) to improve developer experience.

---

### **Conclusion**

- **Yes**, many C++-based databases **support ANSI SQL**, but usually with **extensions and deviations**.
- Compliance levels vary by **database goals**—OLTP (like MySQL/PostgreSQL) follow it more closely than OLAP-focused systems (like ClickHouse).

Let me know if you'd like examples of ANSI SQL compliance in specific databases or feature comparisons.