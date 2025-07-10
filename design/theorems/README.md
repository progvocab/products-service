

Just like the **CAP Theorem**, there are several other **theoretical models and trade-off frameworks** in distributed systems and architecture that are **not design patterns or principles**, but are **used to guide architectural decisions**.

These are usually **laws**, **theorems**, or **axioms** — **constraints** you must consider when designing systems, especially distributed, scalable, or fault-tolerant architectures.

---

## 📚 List of Theoretical Models and Trade-off Theorems

| Theorem / Law                            | Description                                                                                                                                                                                   | Relevance                                                    |
| ---------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| **CAP Theorem**                          | You can choose only 2 of: Consistency, Availability, Partition Tolerance                                                                                                                      | Foundational for distributed databases                       |
| **PACELC Theorem**                       | If Partitioned: choose between Availability or Consistency; Else: choose between Latency or Consistency                                                                                       | Improves on CAP by addressing normal (non-failure) operation |
| **BASE vs ACID**                         | BASE is a softer version of consistency for high availability systems (Basically Available, Soft state, Eventually consistent) vs ACID (Atomic, Consistent, Isolated, Durable) for strict DBs | Helps decide data consistency model                          |
| **FLP Impossibility**                    | In an asynchronous network, no algorithm can guarantee both consensus and termination if even one node fails                                                                                  | Fundamental to consensus protocols (Paxos, Raft)             |
| **Brewer’s Conjecture**                  | The original form of CAP (proved as CAP Theorem)                                                                                                                                              | Academic foundation of CAP                                   |
| **Amdahl's Law**                         | Limits the speedup of parallel systems — adding more nodes doesn’t always help                                                                                                                | Crucial in system scaling                                    |
| **Gustafson's Law**                      | Counters Amdahl's: better scalability if problem size increases with processors                                                                                                               | Useful for big data processing systems                       |
| **Little’s Law**                         | Avg. number of items in a queuing system = Arrival rate × Time in system                                                                                                                      | Used in performance tuning and load estimation               |
| **Peter Principle (System Scalability)** | Systems fail when scaled without revisiting design                                                                                                                                            | Informally used in system complexity discussions             |
| **Byzantine Generals Problem**           | Models communication and trust issues in unreliable networks                                                                                                                                  | Solved by consensus algorithms like PBFT, Raft, Tendermint   |

---

## 🧠 Used For What?

These aren’t design patterns, but they help with:

| Area                                   | Example Models                                |
| -------------------------------------- | --------------------------------------------- |
| **Consistency vs Latency trade-offs**  | CAP, PACELC                                   |
| **Scalability modeling**               | Amdahl’s Law, Gustafson’s Law                 |
| **Queue and load modeling**            | Little’s Law                                  |
| **Fault tolerance / Consensus**        | FLP Impossibility, Byzantine Generals Problem |
| **Reliability in distributed systems** | BASE vs ACID, CAP, PACELC                     |

---

## 🚀 Summary Table

| Name                   | Type          | Concern                                     |
| ---------------------- | ------------- | ------------------------------------------- |
| **CAP**                | Theorem       | Trade-off in distributed systems (C, A, P)  |
| **PACELC**             | Theorem       | Latency vs Consistency when not partitioned |
| **FLP**                | Impossibility | No consensus in async networks with faults  |
| **BASE vs ACID**       | Paradigm      | Consistency models in databases             |
| **Amdahl’s Law**       | Law           | Limits of parallel speedup                  |
| **Gustafson’s Law**    | Law           | Scalability for growing data sizes          |
| **Little’s Law**       | Law           | Queue performance metrics                   |
| **Byzantine Generals** | Problem       | Trust and fault in consensus systems        |

---

## ✅ Not Design Patterns or Principles Because:

* They **don't prescribe reusable code structures** (like Singleton, Factory).
* They **don't offer general design advice** (like SOLID, KISS).
* They are **constraints or mathematical/logical truths** that you **must respect** when designing systems.

---

Would you like diagrams or real system examples (like Kafka, Cassandra, or Kubernetes) mapped to these theorems?
