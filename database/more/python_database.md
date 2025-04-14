There are a few databases written in **Python**, but most databases used in Python applications are written in **C, Java, or other languages** for performance reasons. However, some Python-based databases exist, particularly for embedded use, educational purposes, or specific workloads.  

---

## **🔹 Databases Written in Python**  

### **1️⃣ ZODB (Zope Object Database)**
✔ **Type**: Object-Oriented Database (OODB)  
✔ **Use Case**: Storing Python objects persistently  
✔ **Features**:  
   - Stores native **Python objects** without needing SQL  
   - Supports **transactions** and **versioning**  
   - Works as an **embedded database**  
✔ **Example Use Case**: Used in the **Zope web application framework**  
✔ **Website**: [https://zodb.org/](https://zodb.org/)  

#### **Usage Example (ZODB)**
```python
from ZODB import FileStorage, DB
import transaction

storage = FileStorage.FileStorage("data.fs")
db = DB(storage)
conn = db.open()
root = conn.root()

root['name'] = "Python Database"
transaction.commit()
```

---

### **2️⃣ TinyDB**
✔ **Type**: NoSQL (Document-Oriented, Embedded)  
✔ **Use Case**: Lightweight JSON-based storage for small projects  
✔ **Features**:  
   - **Pure Python** implementation  
   - Stores data as **JSON files**  
   - Supports **querying, indexing, and transactions**  
✔ **Example Use Case**: **Configuration storage, simple apps, IoT**  
✔ **Website**: [https://tinydb.readthedocs.io/](https://tinydb.readthedocs.io/)  

#### **Usage Example (TinyDB)**
```python
from tinydb import TinyDB

db = TinyDB("db.json")
db.insert({"name": "Alice", "age": 25})
print(db.search({"name": "Alice"}))
```

---

### **3️⃣ PickleDB**
✔ **Type**: Key-Value Store (Embedded)  
✔ **Use Case**: Lightweight, dictionary-like database  
✔ **Features**:  
   - **JSON-based storage**  
   - **NoSQL key-value store**  
   - **Very simple API (like Python’s dictionary)**  
✔ **Example Use Case**: **Simple key-value storage for scripts**  
✔ **Website**: [https://github.com/patx/pickledb](https://github.com/patx/pickledb)  

#### **Usage Example (PickleDB)**
```python
import pickledb

db = pickledb.load("data.db", auto_dump=True)
db.set("username", "admin")
print(db.get("username"))  # Output: admin
```

---

### **4️⃣ Buzhug**
✔ **Type**: NoSQL (Index-Based)  
✔ **Use Case**: Faster alternative to SQLite for lightweight applications  
✔ **Features**:  
   - **No SQL required**  
   - Uses **Python object indexing**  
   - Stores structured data  
✔ **Example Use Case**: **Local data storage in Python apps**  
✔ **Website**: [http://buzhug.sourceforge.net/](http://buzhug.sourceforge.net/)  

#### **Usage Example (Buzhug)**
```python
from buzhug import Base

db = Base("mydatabase")
db.create(["name", "age"], mode="overwrite")
db.insert(name="Alice", age=25)
```

---

### **5️⃣ RelationalAI (Python Interface)**
✔ **Type**: AI-Driven Relational Database  
✔ **Use Case**: Querying relational data with **AI-powered optimization**  
✔ **Features**:  
   - **Machine learning-powered query engine**  
   - **Relational algebra approach**  
   - Optimized for **graph analytics**  
✔ **Example Use Case**: **AI-based data analysis**  
✔ **Website**: [https://relational.ai/](https://relational.ai/)  

---

## **🔹 Python Interfaces for Other Databases**
While the above databases are written in Python, Python is mostly used with databases written in **C, C++, or Java**. Popular databases with Python support:  

| **Database** | **Written In** | **Type** | **Python Library** |
|-------------|---------------|----------|---------------------|
| SQLite | C | Relational | `sqlite3` (built-in) |
| PostgreSQL | C | Relational | `psycopg2`, `asyncpg` |
| MySQL | C++ | Relational | `mysql-connector-python`, `PyMySQL` |
| MongoDB | C++ | NoSQL (Document) | `pymongo` |
| Redis | C | NoSQL (Key-Value) | `redis-py` |
| Cassandra | Java | NoSQL (Column) | `cassandra-driver` |

---

## **🚀 Summary: When to Use Python-Based Databases?**
| **Database** | **Best For** |
|-------------|-------------|
| **ZODB** | Persisting Python objects directly |
| **TinyDB** | JSON-based storage for lightweight apps |
| **PickleDB** | Simple key-value storage for scripts |
| **Buzhug** | Indexed storage without SQL |
| **RelationalAI** | AI-driven relational queries |

Most real-world applications, especially at **scale**, use Python with **external databases** (e.g., PostgreSQL, MongoDB, Redis) rather than Python-written databases.

Would you like help with database selection for a specific use case?