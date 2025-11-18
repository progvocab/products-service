
Kafka partitions are subsets of a topic, and each partition is replicated across multiple brokers for fault tolerance. Among these replicas, one broker is elected as the leader, and it handles all reads and writes for that partition. Followers replicate the leader’s data and may temporarily fall out of sync when new messages arrive. Once they catch up to the leader’s latest offset, they rejoin the In-Sync Replica (ISR) set. Because partitions operate independently, Kafka can process messages in parallel across multiple brokers.

Kafka’s high availability comes from the fact that if a leader fails, one of the in-sync follower replicas can be automatically promoted to leader, allowing the partition to continue serving client requests with minimal downtime. The Kafka Controller broker is responsible for managing these leader elections and tracking the ISR set. This architecture ensures both horizontal scalability—by adding more partitions and brokers—and resilience, since data remains available even when individual brokers crash or go offline.


Kafka also ensures efficient data storage and retrieval by organizing partition data into append-only log segments on each broker’s local disk. This design allows for sequential disk writes, which are extremely fast, and lets consumers read at their own pace using their committed offsets. Since consumers track their own progress, multiple consumer groups can independently read the same partition data without impacting broker performance. This log-based structure, combined with replication and partitioning, enables Kafka to handle very high throughput while maintaining durability and replay capability.

Kafka ensures durability by writing messages to the leader partition’s append-only log on disk and replicating them to follower replicas before acknowledging the write to producers. The producer’s acks setting controls how much durability is guaranteed—acks=1 returns once the leader writes to its log, while acks=all returns only after all In-Sync Replicas (ISR) have fully replicated the message. Because ISR members are always caught up to the leader’s latest committed offset, Kafka guarantees that data is safe even if the leader broker fails. This combination of disk persistence, replication, and controlled acknowledgments ensures strong durability and fault tolerance.

In Kafka, the term **“partition”** is what Kafka itself uses.
But **conceptually**, Kafka partitions are **shards** — units of parallelism and scalability.



| Concept                       | Meaning                                                                                  | Example                      |
| ----------------------------- | ---------------------------------------------------------------------------------------- | ---------------------------- |
| **Topic**                     | Logical stream name for a category of messages.                                          | `employee-events`            |
| **Partition (Shard)**         | A subdivision of a topic — each partition is an ordered, immutable sequence of messages. | `partition-0`, `partition-1` |
| **Shard Key / Partition Key** | Key used to decide *which partition* a message goes to.                                  | e.g., `employeeId`           |
| **Broker**                    | Kafka server that stores one or more partitions.                                         | `broker-1`, `broker-2`       |
| **Replication**               | Each partition has replicas on multiple brokers for fault tolerance.                     | Replication factor = 3       |

---

### ⚙️ **How Kafka Uses Sharding (Partitioning)**

1. When a producer sends a message to a topic, it includes a **key**.
2. Kafka uses a **partitioner function** (usually a hash) to map the key to a partition:

   ```
   partition = hash(key) % number_of_partitions
   ```
3. All messages with the same key always go to the same partition → **ordering is preserved per key**.
4. Consumers in the same consumer group each get **one or more partitions** → **parallel consumption**.



```mermaid
flowchart LR
    subgraph Producer
        A1["Message(key=E1)"]
        A2["Message(key=E2)"]
        A3["Message(key=E3)"]
    end

    subgraph "Kafka Cluster"
        subgraph Broker1
            P0["Partition 0 (Shard 0)"]
        end
        subgraph Broker2
            P1["Partition 1 (Shard 1)"]
        end
        subgraph Broker3
            P2["Partition 2 (Shard 2)"]
        end
    end

    subgraph ConsumerGroup
        C1["Consumer 1"]
        C2["Consumer 2"]
        C3["Consumer 3"]
    end

    A1-->|hash(E1)%3=0|P0
    A2-->|hash(E2)%3=1|P1
    A3-->|hash(E3)%3=2|P2

    P0-->|stream events|C1
    P1-->|stream events|C2
    P2-->|stream events|C3
```



| Term                  | In Kafka                                                | Analogy                                       |
| --------------------- | ------------------------------------------------------- | --------------------------------------------- |
| **Shard**             | Partition                                               | Small, independent unit of a topic            |
| **Shard Key**         | Message key                                             | Determines which partition stores the message |
| **Sharding Function** | Hashing function                                        | Maps key → partition                          |
| **Rebalancing**       | Movement of partitions when brokers or consumers change | Automatic load balancing                      |



###  **Dynamic Scaling**

When a **new broker** is added:

* Kafka may **reassign some partitions** to it.
* **Producers** automatically pick up the new metadata.
* **Consumers** rebalance (each partition is processed by exactly one consumer in the group).






A **partition** in Kafka is a **subdivision of a topic** that allows parallelism, fault tolerance, and scalability. Each partition is stored on a Kafka **broker** and is **ordered**.  

- Each message in a partition gets a **unique offset**.  
- **Producers write messages** to partitions.  
- **Consumers read messages** in the order they were produced **within a partition**.  
- Partitions enable **parallel processing** across multiple consumers.  


### **Managing Partitions**  

1. **Choosing the Right Number of Partitions**  
- **More partitions = More parallelism, but too many can hurt performance.**  
- **Too few partitions** → Limits scalability.  
- **Too many partitions** → Increases metadata overhead in Kafka.  


```
Partitions = (Target Throughput per Topic) / (Throughput per Partition)
```
or  
```
Partitions = (Number of Consumers in a Consumer Group) * 2
```
**Best Practice:** Start with **2x the number of consumers** and adjust based on monitoring.  



2. **Creating Partitions**
When creating a topic, specify the number of partitions:  
```bash
bin/kafka-topics.sh --create --topic my_topic --bootstrap-server localhost:9092 --partitions 3 --replication-factor 2
```
🛑 **You cannot decrease partitions** later, but you can **increase** them.



3. **Increasing Partitions for an Existing Topic**
```bash
bin/kafka-topics.sh --alter --topic my_topic --bootstrap-server localhost:9092 --partitions 6
```
🛑 **Caution:** This does not automatically rebalance existing data. Some partitions will have more data than others.



4. **Managing Partition Assignment (Manual or Auto)**
Kafka **automatically** assigns partitions using a **hash function** on the key:  
```python
producer.send("my_topic", key=b"user1", value=b"data")
```
 **Manual Partition Assignment** (if you want full control):  
```python
producer.send("my_topic", value=b"data", partition=1)
```



5. **Rebalancing Partitions (Distribute Load Evenly)**
If brokers are overloaded, use the **Kafka partition reassignment tool**:
```bash
bin/kafka-reassign-partitions.sh --bootstrap-server localhost:9092 --generate
```
This generates a JSON file with new assignments. Apply it:  
```bash
bin/kafka-reassign-partitions.sh --bootstrap-server localhost:9092 --execute --reassignment-json-file reassignment.json
```



### **Best Practices**
✅ **Monitor partition imbalance** using Kafka metrics (`kafka.server:type=ReplicaManager,name=PartitionCount`).  
✅ **Use a key-based partitioning strategy** for consistent ordering per key.  
✅ **Keep the partition count proportional to consumer instances**.  
✅ **Use Kafka Cruise Control** for automatic partition rebalancing.  

