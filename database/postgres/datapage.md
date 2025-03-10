### **🔹 Data Pages in PostgreSQL**  

In **PostgreSQL**, **data pages** are the fundamental storage units where table and index data are stored **on disk**. PostgreSQL **divides tables and indexes into fixed-size pages** (also called blocks), typically **8 KB** in size.

---

## **🔹 How Data Pages Work**
1️⃣ When a row is inserted, updated, or deleted, the data is written to **a page inside a table’s file**.  
2️⃣ PostgreSQL **loads and modifies pages in memory (shared buffers)** before writing back to disk.  
3️⃣ Data pages are managed using **the Buffer Manager**, and changes are recorded in the **Write-Ahead Log (WAL)** for crash recovery.  

---

## **🔹 Structure of a Data Page**
Each data page consists of:
| **Section** | **Description** |
|------------|----------------|
| **Page Header** | Metadata like checksum, LSN (Log Sequence Number) |
| **Item Pointers** | Pointers to tuples (row locations) |
| **Tuple Data** | Actual table rows (heap tuples) |
| **Free Space** | Unused space for future inserts |

🔹 **Example:** If a table has 100 MB of data and each page is **8 KB**, PostgreSQL uses **12,800 pages** (`100MB / 8KB`).

---

## **🔹 How to View Data Pages?**
You can inspect table pages using `pg_filenode_relation()` and `hexdump`:

1️⃣ **Find table file location**
```sql
SELECT pg_relation_filepath('employees');
```

2️⃣ **View raw pages using `pg_pageinspect`**:
```sql
SELECT * FROM pgstattuple('employees');
```
This shows **tuple count, free space, and fragmentation**.

---

## **🔹 Data Page Optimization Techniques**
✅ **Autovacuum & Vacuum** → Cleans up dead tuples and reclaims space.  
✅ **Fillfactor** → Adjusts how full a page should be to **reduce fragmentation** (`ALTER TABLE ... SET (fillfactor=80)`).  
✅ **Cluster & Reindex** → Sorts data pages to **improve query speed**.  
✅ **TOAST Storage** → Stores large data (e.g., `TEXT` and `BYTEA`) outside main pages.  

---

## **🔹 Summary**
| **Concept** | **Description** |
|------------|----------------|
| **Page Size** | Default **8 KB** (configurable) |
| **Data Storage** | Tables & indexes stored in pages |
| **Page Components** | Header, tuples, free space |
| **Performance Tips** | Use `VACUUM`, `FILLFACTOR`, `REINDEX` |

Would you like a **deep dive into page fragmentation and how to fix it?** 🚀