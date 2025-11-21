Here’s a complete, interview-ready explanation of multi-version read consistency (MVCC) in Oracle:


---

MVCC in Oracle

1. Concept:

Oracle uses multi-versioning to provide read consistency.

Each transaction sees a snapshot of the database as of the time the query started.

This ensures readers don’t block writers and writers don’t block readers.



---

2. How it works for operations:

INSERT:

A new row is added with a transaction ID / SCN (System Change Number).

The row is visible to the inserting transaction immediately, but other transactions see it only after commit.


UPDATE:

Oracle keeps the old version of the row in undo segments.

The updated row gets a new SCN.

Other transactions querying the row see the old version until the update transaction commits.


DELETE:

Row is logically marked as deleted with a new SCN.

Other transactions continue to see the old version until commit.


SELECT:

Always sees a consistent snapshot as of query start.

No locks are placed for reads (non-blocking).




---

3. Transaction commit:

When the transaction commits, the undo/redo system updates the SCN, making the new versions visible to other transactions.



---

In short:

MVCC ensures consistent reads without blocking, providing isolation and concurrency.

Each row can have multiple versions until transactions commit.

Inserts, updates, and deletes create new versions, and the CBO/undo system maintains visibility rules.





If you want, I can also make a diagram showing old vs new row versions across concurrent transactions, which is very useful for explaining MVCC in interviews.






1. Inserts create a new row version that is associated with the transaction.


2. After commit, the new version becomes visible to other 

Oracle uses a System Change Number (SCN) internally to track versions,

MVCC is managed via undo segments





For INSERT: there is no old version

For UPDATE/DELETE: Oracle keeps the old version in undo, and the new version has a new SCN.





> When a transaction inserts a row, Oracle creates a new version of the row with an SCN tied to the transaction. The row is visible to the transaction immediately, but other transactions see it only after commit. For updates and deletes, Oracle keeps the old row version in undo segments to provide a consistent read to other transactions until the transaction commits. This ensures read consistency and non-blocking concurrency.


