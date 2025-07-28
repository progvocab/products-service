The **CAP Theorem** describes the trade-offs in distributed systems between **Consistency (C)**, **Availability (A)**, and **Partition Tolerance (P)**. A distributed system can **only guarantee two of the three** at any time:

* **Consistency (C):** Every read receives the most recent write.
* **Availability (A):** Every request receives a response (even if it’s not the latest).
* **Partition Tolerance (P):** The system continues to function despite network partitions.

---

## ⚖️ CAP Theorem Comparison: MongoDB vs PostgreSQL vs Cassandra

Below is a comparison based on CAP characteristics and behavior in distributed environments:

### 🔸 Summary Table:

| Database       | CAP Profile                                           | Explanation                                                                                                                                                                                                  |
| -------------- | ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **PostgreSQL** | **CA** (non-distributed)<br>**CP** (when distributed) | Traditionally single-node: strong consistency and high availability (no partitioning). With tools like Patroni or Citus: CP — consistent and partition-tolerant but may sacrifice availability in partition. |
| **MongoDB**    | **CP** (default) <br> can be tuned to **CA**          | Consistent and partition-tolerant; prioritizes consistency using replica sets. You can trade off consistency for availability using read preferences (`readConcern`, `writeConcern`).                        |
| **Cassandra**  | **AP**                                                | Prioritizes availability and partition tolerance. Uses eventual consistency. Tunable consistency levels allow some balance.                                                                                  |

---

## 🧠 Deeper Breakdown

### ✅ 1. **PostgreSQL** – Traditional RDBMS

* **CAP Mode**:

  * **CA** in standalone mode.
  * **CP** when distributed (e.g., with Patroni, Citus).
* **Behavior**:

  * Strong ACID compliance.
  * Not natively partition-tolerant, but modern setups can distribute it.
* **Use Case**:

  * Financial systems, relational data, joins, strict consistency.

---

### ✅ 2. **MongoDB** – Document Store

* **CAP Mode**:

  * **CP** by default (strong consistency via primary-replica).
  * Can be tuned to **CA** using secondary reads (read preference).
* **Behavior**:

  * Supports tunable **writeConcern** and **readConcern**.
  * During a partition, Mongo will prefer to elect a new primary (may lose availability briefly).
* **Use Case**:

  * Semi-structured data, fast reads/writes, analytics.

---

### ✅ 3. **Cassandra** – Wide-Column Store

* **CAP Mode**: **AP**
* **Behavior**:

  * High availability even under network partitions.
  * Writes always accepted; uses **eventual consistency**.
  * **Tunable consistency** (`QUORUM`, `ONE`, `ALL`) for reads/writes.
* **Use Case**:

  * IoT, telemetry, time-series data, global scale.

---

## 🎯 Choosing Based on CAP

| Requirement                          | Best Fit       |
| ------------------------------------ | -------------- |
| Strong consistency + ACID            | **PostgreSQL** |
| Flexible schema + consistency        | **MongoDB**    |
| High write throughput + availability | **Cassandra**  |

---

Would you like a comparison also involving **availability under failure**, or a visual CAP triangle diagram?
