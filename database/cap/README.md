# CAP

**Cassandra, DynamoDB, MongoDB, Riak, Couchbase, and CosmosDB** are the main databases that *truly offer configurable CAP behavior.*
Spanner and HBase offer partial tuning but are not fully switchable.
**No database gives a literal *“CAP configuration switch”*** — but several databases let you **adjust behavior** so they can operate closer to **CP** or **AP** depending on settings.

Below is the list of databases that **effectively offer configurable CAP trade-offs** through consistency levels, replication modes, or topology choices.

 

### **1. Cassandra**
Highly Configurable , AP by default can be switched to CP
Cassandra provides the most fine-grained control using **tunable consistency**:

* `CONSISTENCY ONE` → AP
* `CONSISTENCY QUORUM` → CP
* `CONSISTENCY ALL` → strongly CP (but low availability)

You configure this **per read and per write**.

### **2. Amazon DynamoDB (Configurable → AP or CP-lite)**

Based on the Dynamo paper; defaults to **AP**.

You can configure:

* **Eventual consistency** → AP
* **Strongly consistent reads** → CP-ish (still P, so reduces availability)

DynamoDB Global Tables = AP by design, but local reads can be strong.

 

### **3. MongoDB**

Configurable , AP by default can be switched to CP
MongoDB allows CAP tuning using:

* **writeConcern** (`1` → AP, `majority` → CP)
* **readConcern** (`local` → AP, `majority` → CP)
* **readPreference** (secondary = AP, primary = CP-ish)

Default: **AP**.
 

### **4. Couchbase (Configurable → CP or AP)**

Modes:

* **Durability Majority Writes** → CP
* **Eventually consistent views/queries** → AP
* **Cross-Datacenter Replication (XDCR)** → AP

 

### **5. Riak (Configurable → CP or AP)**

Fully tunable like Cassandra:

* `r`, `w`, `n` values control consistency vs availability
* e.g., `r=1 w=1` → AP
* ```
  `r=2 w=2` → CP
  ```
 

### **6. Cosmos DB (Highly Configurable with 5 Levels)**

Cosmos DB has **five tunable consistency levels**:

1. Strong → CP
2. Bounded Staleness → almost CP
3. Session → middle
4. Consistent Prefix → AP
5. Eventual → AP

Globally distributed, so CAP behavior depends on selected level.
 

### **7. Google Spanner (Strong CP by default, but tunable for AP regions)**

While Spanner is fundamentally CP (external-commit timestamps), you can:

* Use **read replicas** → AP read scaling
* Use **stale reads** (`AS OF TIMESTAMP`) → AP-ish
 

### **8. HBase (CP by design but configurable)**

HBase is inherently **CP** (strong consistency on row-level), but you can:

* Enable async WAL → slightly AP
* Use replicated tables → eventually consistent reads

Not a full CAP switch, but configurable behavior.
 
 

| Database           | Default Behavior | CAP Configurable?  | How You Configure It                        |
| ------------------ | ---------------- | ------------------ | ------------------------------------------- |
| **Cassandra**      | AP               | **Yes (best)**     | Tunable consistency (QUORUM, ALL, ONE)      |
| **DynamoDB**       | AP               | Yes                | Strong vs eventual reads                    |
| **MongoDB**        | AP               | Yes                | writeConcern + readConcern + readPreference |
| **Couchbase**      | AP               | Yes                | Durability + XDCR modes                     |
| **Riak**           | AP               | Yes                | Tunable consistency (r/w/n)                 |
| **Cosmos DB**      | AP/CP            | **Yes (5 levels)** | Consistency levels                          |
| **Google Spanner** | CP               | Partial            | Stale reads / read replicas                 |
| **HBase**          | CP               | Partial            | WAL + replication settings                  |
 


 

##   **Relational Databases (Oracle, MySQL, PostgreSQL) — CAP Behavior**

Relational databases like **Oracle, MySQL, PostgreSQL, SQL Server** do **NOT** offer configurable CAP behavior.
They inherently operate as **CP systems** when run on a **single node**, and behave as **CA (Consistency + Availability)** in a **traditional non-distributed setup**, but once they are used in a **distributed / replicated / clustered** architecture, their behavior depends on the replication mode — **but still not “configurable CAP”** the way NoSQL systems support.

### **1. On a Single Node (Traditional Deployment)**

A single-node RDBMS is not a distributed system, CAP theorem does not apply.
However, conceptually:

| Relational DB | Consistency | Availability | Partition Tolerance             | CAP Category |
| ------------- | ----------- | ------------ | ------------------------------- | ------------ |
| Oracle        | ✔ Strong    | ✔ High       |   No network partition scenario | **CA**       |
| MySQL         | ✔ Strong    | ✔ High       |                                 | **CA**       |
| PostgreSQL    | ✔ Strong    | ✔ High       |                                 | **CA**       |

 

### **2. When Used in Distributed Replication / Clustering**

RDBMSs enforce **strong consistency first**, so in distributed mode they lean toward **CP**.

### **Why CP?**

* They prefer **consistency** over availability
* If a network partition happens, they will **reject writes** to avoid inconsistency
 
 

### **Oracle RAC**

* Shared cache architecture → **strong consistency**
* In partitions, RAC may **freeze or fail sessions** to avoid split brain
    **Oracle RAC = CP**

### **Oracle Data Guard**

Depending on configuration:

| Mode                 | Behavior                                        | CAP                                                          |
| -------------------- | ----------------------------------------------- | ------------------------------------------------------------ |
| Maximum Protection   | Never lose a transaction                        | CP                                                           |
| Maximum Availability | Prefers consistency but tries to stay available | CP Leaning                                                   |
| Maximum Performance  | Fast async replica                              | **AP-leaning**, but *not configurable CAP* (still mostly CP) |

  *Oracle never allows you to choose "AP mode" fully, because it prevents divergent writes.*

 

### **MySQL Group Replication**

Modes:

* **Single-Primary Mode** → strong consistency → CP
* **Multi-Primary Mode** → conflicts possible → AP‐leaning
  but still not a formal AP system like Cassandra.

### **MySQL Async Replication**

* The replica can lag
* Allows writes even when replicas unreachable
    This is **AP-leaning**, but **not configurable as CAP modes**.
 You *cannot* configure MySQL like Cassandra (“choose AP or CP”).

 

### **PostgreSQL Streaming Replication**

* Sync Mode → CP
* Async Mode → AP-leaning
  (replica may lose last transactions)

### **Patroni / etcd HA setups**

* Leader election ensures consistency
 **PostgreSQL HA = CP**
 

| Database   | CP | AP                                  | Configurable CAP Mode Available? |
| ---------- | -- | ----------------------------------- | -------------------------------- |
| Oracle     | Yes  | Partial-AP in async replication     |   No                             |
| MySQL      | Yes  | Partial-AP in async / multi-primary |   No                             |
| PostgreSQL | Yes  | Partial-AP in async replication     |   No                             |
| SQL Server | Yes  | Partial-AP                          |   No                             |

 **Relational DBs are fundamentally CP systems.
They do NOT allow configurable CAP behavior like Cassandra or Cosmos DB.**

 

| Database                      | Supports Switching CP/AP?                                               | Notes                                             |
| ----------------------------- | ----------------------------------------------------------------------- | ------------------------------------------------- |
| **Microsoft Azure Cosmos DB** | ✔ Yes (strong, bounded staleness, session, consistent prefix, eventual) | Only DB with full consistency-level configuration |
| **Cassandra**                 | ✔ Tunable (via consistency levels)                                      | per-query consistency (ALL, QUORUM, ONE)          |
| **Riak**                      | ✔ Tunable                                                               | AP by default                                     |
| **DynamoDB**                  | ✔ Per-request strong/eventual                                           | limited but configurable                          |
| **Couchbase**                 | Partial                                                                 | stronger consistency with N1QL                    |

 

More : 

 
* A **decision guide: which DB to choose for CP vs AP workloads**
* A **table mapping each DB to its consistency configuration parameters**
* A full CAP comparison matrix for **all major databases**
* CAP behavior diagrams
* Tunable consistency examples (Cassandra, Cosmos, DynamoDB)
 



