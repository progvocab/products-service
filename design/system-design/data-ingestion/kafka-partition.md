### **Enhancing Kafka Consumer Scalability & Efficient Partitioning Strategy**

Since your system deals with **high-volume IoT data**, we need:
1. **Kafka Partitioning Strategy** to distribute data efficiently across multiple consumers.
2. **Scalable Kafka Consumers** using **multiple instances in a Consumer Group**.

---

## **1. Kafka Partitioning Strategy**
Kafka topics support **partitions**, allowing multiple consumers to read in parallel. We will:
✅ **Partition by Device ID**: Ensures all data from a device goes to the same partition.  
✅ **Use Keyed Partitioning**: Kafka routes messages with the same key (Device ID) to the same partition.

### **Kafka Topic Configuration**
```sh
kafka-topics.sh --create --topic access_events --bootstrap-server localhost:9092 --partitions 6 --replication-factor 3
```
Here, we create **6 partitions** to allow multiple consumers to process data in parallel.

---

## **2. Kafka Producer with Partitioning (Go)**
Modify the producer to use **device_id as the partition key**.
```go
// Produce message to Kafka with Keyed Partitioning
err := s.producer.Produce(&kafka.Message{
    TopicPartition: kafka.TopicPartition{Topic: &"access_events", Partition: kafka.PartitionAny},
    Key:            []byte(event.DeviceId), // Partition by device_id
    Value:          []byte(message),
}, nil)
```
Kafka will hash **device_id** and send messages to the same partition.

---

## **3. Scalable Kafka Consumer (Python)**
Consumers must be in a **Consumer Group** so they can **scale dynamically**.

### **Kafka Consumer in Consumer Group**
```python
from kafka import KafkaConsumer
import boto3
import json
import os

# Kafka Config
consumer = KafkaConsumer(
    'access_events',
    bootstrap_servers='localhost:9092',
    group_id='iot-consumers',  # Consumer Group for scaling
    enable_auto_commit=True,
    value_deserializer=lambda x: x.decode('utf-8')
)

# AWS S3 Setup
s3 = boto3.client('s3')

bucket_name = "your-s3-raw-bucket"

def process_message(message):
    """Process and upload data to S3"""
    data = json.loads(message)
    file_key = f"access_logs/{data['device_id']}_{data['timestamp']}.json"
    s3.put_object(Bucket=bucket_name, Key=file_key, Body=json.dumps(data))
    print(f"Stored event in S3: {file_key}")

# Consume Messages
for msg in consumer:
    process_message(msg.value)
```
### **Why this is scalable?**
✅ Multiple consumers in **same group ("iot-consumers")** distribute load dynamically  
✅ Kafka assigns partitions to consumers automatically  
✅ If a consumer crashes, Kafka **reassigns partitions** to available consumers  

---

## **4. Scaling Consumers in Kubernetes**
### **Step 1: Create a Kubernetes Deployment**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: kafka-consumer
spec:
  replicas: 4  # Scale to 4 consumers
  selector:
    matchLabels:
      app: kafka-consumer
  template:
    metadata:
      labels:
        app: kafka-consumer
    spec:
      containers:
        - name: kafka-consumer
          image: myrepo/kafka-consumer:latest
          env:
            - name: KAFKA_BROKER
              value: "kafka-service:9092"
            - name: KAFKA_TOPIC
              value: "access_events"
            - name: AWS_BUCKET
              value: "your-s3-raw-bucket"
```
### **Step 2: Enable Horizontal Scaling**
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: kafka-consumer-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: kafka-consumer
  minReplicas: 3
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
```
✅ **Autoscaling (HPA)**: Increases consumer count when CPU usage is high.  
✅ **Multiple Replicas**: Each consumer reads from different partitions.

---

## **5. Summary**
### **Kafka Partitioning**
- **Partition by `device_id`** for efficient distribution.
- **Increase Kafka partitions** to handle high throughput.

### **Kafka Consumers**
- Use **Consumer Groups** to allow auto-scaling.
- Deploy multiple consumer instances in **Kubernetes**.

### **Auto-Scaling**
- **Horizontal Scaling (HPA)**: Adds more consumers as needed.
- **Fault Tolerance**: If one consumer dies, Kafka rebalances partitions.

Would you like help with **deploying Kafka & Kubernetes in AWS (EKS/MSK)?**