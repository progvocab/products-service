### **How to Add More Brokers to a Kafka Cluster**  

Scaling a Kafka cluster by **adding more brokers** increases fault tolerance, improves performance, and distributes the load more efficiently.  

---

## **🔹 Steps to Add More Brokers to a Kafka Cluster**  

### **1️⃣ Install Kafka on the New Broker Machine**
If the new broker is on a **new server**, install Kafka:  
```bash
wget https://downloads.apache.org/kafka/3.5.0/kafka_2.13-3.5.0.tgz
tar -xvf kafka_2.13-3.5.0.tgz
cd kafka_2.13-3.5.0
```

---

### **2️⃣ Configure the New Broker (`server.properties`)**  
Each broker must have a **unique broker ID** and be aware of the existing cluster.  

Edit the **Kafka configuration file** on the new broker:  
```bash
nano config/server.properties
```
Update or add the following properties:  

```properties
broker.id=2   # Must be unique across all brokers
log.dirs=/var/lib/kafka-logs  # Path to store Kafka data
zookeeper.connect=zk-host:2181  # Ensure it points to the same ZooKeeper as other brokers
listeners=PLAINTEXT://:9093  # Ensure this port is unique on this machine
advertised.listeners=PLAINTEXT://new-broker-ip:9093
```
---
  
### **3️⃣ Start the New Broker**  
Run Kafka with the updated config:  
```bash
bin/kafka-server-start.sh config/server.properties &
```
Verify that the broker has started successfully by checking logs:  
```bash
cat logs/server.log | grep "started"
```
---

### **4️⃣ Confirm the New Broker is Added**  
On any existing broker, run:  
```bash
bin/kafka-broker-api-versions.sh --bootstrap-server existing-broker-ip:9092
```
You should see the new broker **listed in the output**.  

Alternatively, check using **ZooKeeper**:  
```bash
bin/zookeeper-shell.sh zk-host:2181 <<< "ls /brokers/ids"
```
---

### **5️⃣ Reassign Partitions to Utilize the New Broker (Optional)**
Kafka does not **automatically move partitions** to new brokers. You need to manually reassign partitions to balance the cluster.  

#### **Step 1: Generate the Reassignment Plan**
```bash
bin/kafka-reassign-partitions.sh --zookeeper zk-host:2181 --generate
```
It will output a JSON file suggesting partition movements.

#### **Step 2: Execute the Reassignment**
```bash
bin/kafka-reassign-partitions.sh --zookeeper zk-host:2181 --execute --reassignment-json-file reassignment.json
```

#### **Step 3: Verify Reassignment**
```bash
bin/kafka-reassign-partitions.sh --zookeeper zk-host:2181 --verify --reassignment-json-file reassignment.json
```
---

## **🔹 Summary**
1. **Install Kafka** on the new machine.  
2. **Update `server.properties`** with a new `broker.id` and proper configurations.  
3. **Start the new broker** and verify it in the cluster.  
4. **Reassign partitions** to distribute the load evenly (optional).  

---

## **🔹 Best Practices**
✅ Use **Kafka's built-in metrics** to monitor load before adding brokers.  
✅ Distribute partitions **evenly** across brokers.  
✅ If using **Kafka without ZooKeeper (KRaft mode)**, use `bin/kafka-metadata-shell.sh` to verify broker membership.  

Would you like guidance on **automatic partition rebalancing** with Cruise Control?