### zookeeper
config file :
/etc/kafka/zookeeper.properties
![image](https://github.com/user-attachments/assets/9998eab4-785d-45ae-a452-11e9eef5eb7b)



Apache Kafka uses **ZooKeeper** (in versions before Kafka 2.8 for cluster coordination) to manage broker metadata, topic configurations, partition assignments, and controller elections. While most Kafka management is done using Kafka’s own CLI tools, these tools often **interact with ZooKeeper under the hood**.

Here are the **commonly used Kafka commands** that **require ZooKeeper**, and are used to manage Kafka:

---

### 🔧 **Common Kafka Commands Using ZooKeeper**

> 🧠 **Replace** `localhost:2181` with your actual ZooKeeper connect string.

#### ✅ **1. List all Kafka topics**

```bash
kafka-topics.sh --list --zookeeper localhost:2181
```

#### 📝 **2. Create a new topic**

```bash
kafka-topics.sh --create \
  --zookeeper localhost:2181 \
  --replication-factor 1 \
  --partitions 3 \
  --topic my-topic
```

#### ✏️ **3. Alter topic configuration (e.g., increase partitions)**

```bash
kafka-topics.sh --alter \
  --zookeeper localhost:2181 \
  --topic my-topic \
  --partitions 5
```

#### ❌ **4. Delete a topic**

```bash
kafka-topics.sh --delete \
  --zookeeper localhost:2181 \
  --topic my-topic
```

> ⚠️ The topic must be marked for deletion in broker config (`delete.topic.enable=true`).

#### 📋 **5. Describe a topic**

```bash
kafka-topics.sh --describe \
  --zookeeper localhost:2181 \
  --topic my-topic
```

---

### 👥 **Consumer Group Commands (Old API using ZooKeeper)**

Before Kafka 0.10.x, consumers stored offsets in ZooKeeper.

#### 📋 **List consumer groups (older style)**

```bash
kafka-consumer-groups.sh --zookeeper localhost:2181 --list
```

#### 📋 **Describe a specific consumer group**

```bash
kafka-consumer-groups.sh --zookeeper localhost:2181 --describe --group my-group
```

> ⚠️ **Newer consumer groups use `--bootstrap-server`** and no longer depend on ZooKeeper.

---

### 🏁 **Leader Election (Manual Reassignments)**

#### 🔁 **Generate a partition reassignment JSON**

```bash
kafka-reassign-partitions.sh \
  --zookeeper localhost:2181 \
  --topics-to-move-json-file topics.json \
  --broker-list "1,2" \
  --generate
```

#### 🚀 **Execute a reassignment**

```bash
kafka-reassign-partitions.sh \
  --zookeeper localhost:2181 \
  --reassignment-json-file reassignment.json \
  --execute
```

#### 🔍 **Verify reassignment**

```bash
kafka-reassign-partitions.sh \
  --zookeeper localhost:2181 \
  --reassignment-json-file reassignment.json \
  --verify
```

---

### ⚠️ **Important Notes**

1. **Kafka 2.8+ can run without ZooKeeper** — with the **KRaft (Kafka Raft)** mode.
2. All newer tooling prefers `--bootstrap-server` instead of `--zookeeper`.
3. You can also access ZooKeeper directly using tools like `zkCli.sh`, but this is not recommended unless you're debugging.

---

### 🐘 **ZooKeeper CLI (Advanced / Debug)**

To explore ZooKeeper paths used by Kafka:

```bash
zkCli.sh -server localhost:2181
```

Then run commands like:

```bash
ls /brokers/ids
ls /config/topics
get /brokers/ids/0
```

These show broker metadata, topic configs, etc.

---

Would you like a cheat sheet or bash script with these commands pre-filled?

