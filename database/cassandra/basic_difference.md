# **DynamoDB vs Cassandra: Similarities & Differences with Code Examples**  

Amazon DynamoDB and Apache Cassandra are both **highly scalable NoSQL databases** designed for distributed workloads. However, they differ in **deployment, architecture, consistency, and pricing**. Below is a **detailed comparison** with **code examples**.

---

## **🚀 Key Similarities**
| Feature | DynamoDB | Cassandra |
|---------|----------|-----------|
| **NoSQL Database** | ✅ Yes | ✅ Yes |
| **Distributed Architecture** | ✅ Yes | ✅ Yes |
| **Horizontal Scalability** | ✅ Yes | ✅ Yes |
| **Write-Optimized** | ✅ Yes | ✅ Yes |
| **Eventual Consistency** | ✅ Yes (by default) | ✅ Yes (default) |
| **Multi-Region Replication** | ✅ Yes (Global Tables) | ✅ Yes (Multi-DC Support) |

---

## **❌ Key Differences**
| Feature | DynamoDB | Cassandra |
|---------|----------|-----------|
| **Deployment** | Managed (AWS Only) | Self-hosted / Managed (Any Cloud) |
| **Query Language** | **DynamoDB API** | **CQL (Cassandra Query Language)** |
| **Consistency** | **Eventual (default), Strong (optional)** | **Eventual (default), Tunable** |
| **Performance** | Auto-scaled, pay-as-you-go | Predictable but needs manual scaling |
| **Cost Model** | Pay-per-use | Self-managed (fixed infra cost) |
| **Secondary Indexes** | Global & Local Indexes | Materialized Views & Secondary Indexes |

---

## **1️⃣ Data Model Comparison**
### **DynamoDB (JSON-Based Model)**
DynamoDB uses a **key-value store** with flexible JSON attributes.

```json
{
  "UserID": "123",
  "Name": "John Doe",
  "Email": "john@example.com",
  "Orders": [
    {"OrderID": "001", "Amount": 50},
    {"OrderID": "002", "Amount": 30}
  ]
}
```

### **Cassandra (Wide-Column Model)**
Cassandra stores data in a **column-family structure**.

```sql
CREATE TABLE users (
    user_id UUID PRIMARY KEY,
    name TEXT,
    email TEXT
);
```

---

## **2️⃣ Creating a Table**
### **DynamoDB (AWS SDK - Python)**
```python
import boto3

dynamodb = boto3.resource("dynamodb")

table = dynamodb.create_table(
    TableName="Users",
    KeySchema=[{"AttributeName": "UserID", "KeyType": "HASH"}],
    AttributeDefinitions=[{"AttributeName": "UserID", "AttributeType": "S"}],
    BillingMode="PAY_PER_REQUEST"
)
```
🔹 **DynamoDB uses JSON-like key-value structure**  
🔹 **No predefined schema required**  

### **Cassandra (CQL - SQL-like)**
```sql
CREATE TABLE users (
    user_id UUID PRIMARY KEY,
    name TEXT,
    email TEXT
);
```
🔹 **Cassandra uses a structured schema with predefined columns**  
🔹 **Similar to SQL but optimized for distributed storage**  

---

## **3️⃣ Inserting Data**
### **DynamoDB (AWS SDK - Python)**
```python
table.put_item(
    Item={
        "UserID": "123",
        "Name": "John Doe",
        "Email": "john@example.com"
    }
)
```

### **Cassandra (CQL)**
```sql
INSERT INTO users (user_id, name, email) VALUES (uuid(), 'John Doe', 'john@example.com');
```

---

## **4️⃣ Querying Data**
### **DynamoDB: Query with Primary Key**
```python
response = table.get_item(Key={"UserID": "123"})
print(response["Item"])
```
🔹 **DynamoDB requires a primary key for efficient lookups**  
🔹 **Supports global and local secondary indexes for filtering**  

### **Cassandra: Query by Primary Key**
```sql
SELECT * FROM users WHERE user_id = 123;
```
🔹 **Cassandra allows primary key queries**  
🔹 **Supports secondary indexes but with some performance trade-offs**  

---

## **5️⃣ Scalability & Performance**
| Feature | DynamoDB | Cassandra |
|---------|----------|-----------|
| **Scalability** | Automatic Scaling (AWS scales for you) | Manual Scaling (Add nodes as needed) |
| **Read/Write Performance** | Serverless, pay-as-you-go | Linearly scalable, predictable cost |
| **Latency** | Low for small queries, but **variable due to autoscaling** | Consistent performance, **better for large datasets** |

---

## **6️⃣ Multi-Region Replication**
### **DynamoDB Global Tables (AWS Managed)**
```python
dynamodb.create_table(
    TableName="Users",
    KeySchema=[{"AttributeName": "UserID", "KeyType": "HASH"}],
    AttributeDefinitions=[{"AttributeName": "UserID", "AttributeType": "S"}],
    BillingMode="PAY_PER_REQUEST",
    ReplicationGroup=[{"RegionName": "us-west-2"}, {"RegionName": "eu-central-1"}]
)
```
🔹 **AWS automatically handles replication**  
🔹 **Built-in conflict resolution**  

### **Cassandra Multi-Datacenter Replication**
```sql
ALTER KEYSPACE mykeyspace WITH replication = {
  'class': 'NetworkTopologyStrategy',
  'us-east': 3,
  'us-west': 3
};
```
🔹 **Cassandra requires manual configuration**  
🔹 **Greater flexibility but requires maintenance**  

---

## **7️⃣ Cost Considerations**
| **Factor** | **DynamoDB** | **Cassandra** |
|-----------|-------------|--------------|
| **Pricing Model** | Pay-per-use (AWS bills per read/write) | Fixed cost (run your own cluster) |
| **Storage Costs** | AWS charges for storage and read/write units | Self-hosted (disk & infra costs) |
| **Best for** | **Spiky workloads, startups, auto-scaling needs** | **Predictable workloads, large datasets** |

---

## **🎯 When to Choose What?**
| **Use Case** | **Best Choice** |
|-------------|----------------|
| **Serverless, managed, pay-as-you-go** | ✅ DynamoDB |
| **Self-hosted, large datasets, predictable cost** | ✅ Cassandra |
| **Multi-region replication with auto-scaling** | ✅ DynamoDB |
| **Highly write-intensive workloads with predictable latency** | ✅ Cassandra |
| **Deep query support (SQL-like structure)** | ✅ Cassandra |
| **AWS ecosystem integration** | ✅ DynamoDB |

---

## **🚀 Final Thoughts**
Both **DynamoDB and Cassandra** are powerful NoSQL databases, but they serve different purposes:
- **DynamoDB** is best for **serverless, managed solutions with auto-scaling**.
- **Cassandra** is best for **self-hosted, predictable workloads with massive write operations**.

Would you like a **benchmark performance test** or a **real-world case study comparison**? 🚀