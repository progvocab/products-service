Here’s a complete **system design** and **implementation** for your use case:

---

## **System Overview**
### **Components:**
1. **IoT Devices**: Emit access data (entry/exit events) via gRPC.
2. **gRPC Server (Go)**: Receives data from IoT devices and pushes it to **Kafka**.
3. **Kafka Producer (Go)**: Publishes events to a Kafka topic.
4. **Kafka Consumer (Python/Go)**: Reads events from Kafka and stores them in **AWS S3 (Raw Bucket)**.

---

## **1. Define the gRPC Service**
Create a **gRPC proto file (`access.proto`)**:
```proto
syntax = "proto3";

package access;

service AccessService {
  rpc SendAccessEvent (AccessEvent) returns (Ack);
}

message AccessEvent {
  string device_id = 1;
  string user_id = 2;
  string action = 3; // "IN" or "OUT"
  string timestamp = 4;
}

message Ack {
  string message = 1;
}
```
- **`SendAccessEvent`**: IoT devices call this to send data.
- **`AccessEvent`**: Contains details like **device ID, user ID, action (IN/OUT), timestamp**.

---

## **2. Implement the gRPC Server (Go)**
### **Install Dependencies**
```sh
go mod init grpc-kafka
go get google.golang.org/grpc google.golang.org/protobuf github.com/confluentinc/confluent-kafka-go/kafka
```

### **Server Implementation (`server.go`)**
```go
package main

import (
	"context"
	"fmt"
	"log"
	"net"

	"github.com/confluentinc/confluent-kafka-go/kafka"
	pb "path/to/generated/access/proto"

	"google.golang.org/grpc"
)

type AccessServer struct {
	pb.UnimplementedAccessServiceServer
	producer *kafka.Producer
}

// gRPC function to handle incoming data
func (s *AccessServer) SendAccessEvent(ctx context.Context, event *pb.AccessEvent) (*pb.Ack, error) {
	message := fmt.Sprintf("%s,%s,%s,%s", event.DeviceId, event.UserId, event.Action, event.Timestamp)

	// Send data to Kafka topic
	err := s.producer.Produce(&kafka.Message{
		TopicPartition: kafka.TopicPartition{Topic: &"access_events", Partition: kafka.PartitionAny},
		Value:          []byte(message),
	}, nil)

	if err != nil {
		return nil, err
	}

	log.Printf("Received event: %s", message)
	return &pb.Ack{Message: "Event Received"}, nil
}

func main() {
	// Initialize Kafka producer
	producer, err := kafka.NewProducer(&kafka.ConfigMap{"bootstrap.servers": "localhost:9092"})
	if err != nil {
		log.Fatal(err)
	}

	// Start gRPC server
	lis, err := net.Listen("tcp", ":50051")
	if err != nil {
		log.Fatal(err)
	}

	grpcServer := grpc.NewServer()
	pb.RegisterAccessServiceServer(grpcServer, &AccessServer{producer: producer})

	log.Println("gRPC Server listening on port 50051...")
	if err := grpcServer.Serve(lis); err != nil {
		log.Fatal(err)
	}
}
```
---

## **3. Kafka Consumer to Push Data to S3 (Python)**
### **Install Dependencies**
```sh
pip install kafka-python boto3
```

### **Consumer Implementation (`consumer.py`)**
```python
from kafka import KafkaConsumer
import boto3
import json

# Kafka setup
consumer = KafkaConsumer(
    'access_events',
    bootstrap_servers='localhost:9092',
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    value_deserializer=lambda x: x.decode('utf-8')
)

# AWS S3 setup
s3 = boto3.client('s3', aws_access_key_id='YOUR_ACCESS_KEY', aws_secret_access_key='YOUR_SECRET_KEY')
bucket_name = "your-s3-raw-bucket"

def process_message(message):
    # Convert message to JSON
    data = {
        "device_id": message.split(",")[0],
        "user_id": message.split(",")[1],
        "action": message.split(",")[2],
        "timestamp": message.split(",")[3]
    }

    # Upload JSON to S3
    file_key = f"access_logs/{data['device_id']}_{data['timestamp']}.json"
    s3.put_object(Bucket=bucket_name, Key=file_key, Body=json.dumps(data))

    print(f"Stored event in S3: {file_key}")

# Consume messages
for msg in consumer:
    process_message(msg.value)
```
---

## **4. Deployment Considerations**
### **Scalability**
✅ **gRPC Load Balancer**: Deploy multiple instances behind **AWS ALB or Nginx**  
✅ **Kafka Cluster**: Use **multi-node Kafka (AWS MSK)**  
✅ **S3 Lifecycle Rules**: Set retention policies for raw data  

### **Security**
✅ **IAM Roles**: Secure **S3 access** for Kafka consumers  
✅ **TLS Encryption**: Enable **gRPC with TLS**  

---

## **5. Summary**
✅ **IoT Devices** → Send access events via **gRPC**  
✅ **Spring Boot gRPC Server** → Receives data & pushes to **Kafka**  
✅ **Kafka Consumer (Python)** → Reads data & uploads to **S3**  
✅ **AWS S3** → Stores raw access logs  

Would you like help with **deploying on AWS EKS**?