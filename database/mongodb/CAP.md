# CAP
Consitancy,Availability and Partition Tolerence

**MongoDB does *not* let you “configure CAP” directly**, but it *does* let you **choose trade-offs that map to CAP** through write concerns, read concerns, and replica-set architecture.


In a distributed system, you can only guarantee **two** of the three:

* **C — Consistency**
* **A — Availability**
* **P — Partition Tolerance** (always required in distributed systems)

Because MongoDB is a distributed, sharded, replica-set database, it **must support P**.
So in practice MongoDB lets you choose **Consistency (CP)** or **Availability (AP)** behaviors.
**MongoDB’s default behavior is closer to *AP* (Availability + Partition Tolerance)**,  

###   MongoDB Defaults to AP

MongoDB by default uses:

* **writeConcern: 1**
  (acknowledge write only from primary, not majority → allows availability but not strict consistency)

* **readConcern: "local"**
  (reads from the primary’s local state, not majority-committed → can see unreplicated data)

* **readPreference: primary**
  (helps consistency, but not enough to make the system CP)

These defaults ensure the replica set remains **highly available**, even if replication lag or partial network partitions exist.

MongoDB acts as AP , prioritize availability over correctness.




Even though **reads default to primary**, which looks consistent, MongoDB is **not CP by default** because:

* Writes are **not majority acknowledged**
* Reads are **not guaranteed majority-consistent**
* Partitions or failover events can cause clients to read stale or divergent data

Thus default behavior aligns with **AP**.

Default Configuration 

* **writeConcern: 1**
* **readConcern: "local"**
* **Read preference: secondary or nearest**
* **Allow secondary reads even if replication lag exists**

Effects:

* Writes succeed even with limited replica acknowledgment
* Reads may return stale data
* System stays available even during partitions

MongoDB prioritizes **availability over consistency**.

## Change to CP Mode (Consistency + Partition Tolerance)

Only when you change:

* **writeConcern: "majority"**
* **readConcern: "majority"**

Then MongoDB behaves like a **CP system**—preferring consistency and stopping writes during partition/failover.

 




MongoDB does **not have a CAP toggle**, but its **read concern**, **write concern**, and **replica-set topology** let you move closer to CP or AP.

MongoDB acts as CP when you use strong consistency settings.

#### Configure for CP:

* **writeConcern: "majority"**
* **readConcern: "majority"**
* **Read preference: primary**
* **Journaling: enabled**

In this case:

* Writes must be acknowledged by a majority → consistent
* Reads are taken from primary → consistent
* If primary goes down, writes stop → reduced availability

MongoDB prioritizes **consistency over availability**.





| Mode                                        | How to Configure                                                 | Behavior                                                                      |
| ------------------------------------------- | ---------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| **CP** (Consistency + Partition Tolerance)  | majority writeConcern + majority readConcern + read from primary | Consistent reads, consistent writes, but reduced availability during failover |
| **AP** (Availability + Partition Tolerance) | writeConcern: 1, readConcern: local, readPreference: secondary   | Always available, but stale reads possible                                    |
| **P** (Partition Tolerance)                 | Always present in MongoDB                                        | Distributed system requirement                                                |



 

MongoDB **does not explicitly expose CAP settings**, by default it is **AP** but through **read/write concerns and replica-set configuration**, you can **tune MongoDB to behave as CP**.


More: how MongoDB replication & elections affect CAP

