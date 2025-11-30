# BASE

**BASE** is a consistency model used in **NoSQL / distributed databases** as an alternative to the strict **ACID** model used in relational databases.

### What BASE Stands For

**B — Basically Available**
System guarantees availability even during failures.
Reads/writes may still succeed even if some nodes are down.

**A — Soft State**
The state of the system may not be immediately consistent.
Data can change "in the background" due to replication, hints, anti-entropy, etc.

**SE — Eventually Consistent**
All replicas will become consistent *eventually*, but not immediately.
There is no guarantee *when* they will converge.

### When BASE Is Used

* Distributed systems
* Large-scale NoSQL databases
* High-availability systems
* Use cases where speed and availability matter more than strict consistency

### Examples of BASE Databases

* **Cassandra**
* **DynamoDB**
* **Riak**
* **Couchbase**

### BASE vs ACID (Simple Comparison)

| ACID                      | BASE                                 |
| ------------------------- | ------------------------------------ |
| Strict consistency        | Eventual consistency                 |
| Transactions always valid | Data may be temporarily inconsistent |
| Strong guarantees         | High availability                    |
| Used in RDBMS             | Used in NoSQL                        |

### Key Idea

BASE sacrifices **immediate consistency** to gain:

* Scalability
* High availability
* Partition tolerance

If you want, I can also give you a **Mermaid diagram** comparing ACID vs BASE or deep internal architecture examples (like how Cassandra implements eventual consistency).
