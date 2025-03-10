### **🔹 AWS Glue Job: Incremental Data Extraction from MongoDB to Data Lake (S3)**  

This AWS Glue job will **incrementally extract** employee data from MongoDB and store it in an **S3 data lake**.  

---

## **🔹 Steps to Implement:**
1️⃣ **Set Up MongoDB Connection**  
2️⃣ **Extract Incremental Data** (Based on `last_modified` timestamp)  
3️⃣ **Transform Data (if needed)**  
4️⃣ **Write Data to S3 in Parquet Format**  
5️⃣ **Store State for Next Incremental Run**  

---

## **1️⃣ Prerequisites**  
✅ **MongoDB**: Must have a `last_modified` field for incremental extraction.  
✅ **AWS Glue Connection**: Create a **JDBC connection** to MongoDB.  
✅ **IAM Role**: Grant Glue access to S3 and MongoDB.  
✅ **S3 Data Lake**: Predefine an S3 bucket (`s3://my-data-lake/employees/`).  
✅ **AWS Glue Catalog**: Define schema for query access.  

---

## **2️⃣ AWS Glue Script (Python - PySpark)**
```python
import sys
from awsglue.transforms import *
from awsglue.dynamicframe import DynamicFrame
from awsglue.job import Job
from awsglue.context import GlueContext
from pyspark.context import SparkContext
from datetime import datetime
import boto3

# Initialize Glue context
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)

# Define MongoDB Connection
mongo_uri = "mongodb://username:password@mongo-host:27017/employees_db.employees"

# Define S3 Target Path
s3_target_path = "s3://my-data-lake/employees/"

# Read Last Processed Timestamp from S3 (Incremental Load Tracking)
s3_client = boto3.client('s3')
state_file = "s3://my-data-lake/state/last_sync_timestamp.txt"

try:
    obj = s3_client.get_object(Bucket="my-data-lake", Key="state/last_sync_timestamp.txt")
    last_sync_timestamp = obj['Body'].read().decode('utf-8')
except:
    last_sync_timestamp = "1970-01-01T00:00:00Z"  # Default for first run

print(f"Last processed timestamp: {last_sync_timestamp}")

# Extract Incremental Data from MongoDB
mongo_query = {'last_modified': {'$gt': last_sync_timestamp}}
df = spark.read.format("com.mongodb.spark.sql.DefaultSource") \
    .option("uri", mongo_uri) \
    .option("pipeline", f"[{{'$match': {mongo_query}}}]") \
    .load()

if df.count() > 0:
    # Convert to Glue DynamicFrame
    dynamic_frame = DynamicFrame.fromDF(df, glueContext, "dynamic_frame")

    # Write Data to S3 in Parquet Format (Incremental)
    glueContext.write_dynamic_frame.from_options(
        frame=dynamic_frame,
        connection_type="s3",
        connection_options={"path": s3_target_path, "partitionKeys": ["department"]},
        format="parquet"
    )

    # Update Last Sync Timestamp
    new_last_sync_timestamp = df.agg({"last_modified": "max"}).collect()[0][0]

    # Save new timestamp in S3 for next run
    s3_client.put_object(
        Bucket="my-data-lake",
        Key="state/last_sync_timestamp.txt",
        Body=new_last_sync_timestamp
    )
    print(f"Updated last processed timestamp: {new_last_sync_timestamp}")

else:
    print("No new data to process.")

# Commit Glue Job
job.commit()
```

---

## **3️⃣ Key Features in This Script**
✅ **Incremental Extraction**: Reads only new/updated records since the last sync.  
✅ **Efficient Storage**: Writes data to S3 in **Parquet format** (optimized for analytics).  
✅ **State Management**: Stores the last processed timestamp in **S3 state file**.  
✅ **Partitioning**: Partitions by `department` for better query performance.  

---

## **4️⃣ How to Deploy in AWS Glue**
### **Step 1: Create an AWS Glue Job**
1. Go to **AWS Glue Console** → **Jobs** → **Create Job**.  
2. Choose **"Spark (Python)"** as the Glue job type.  
3. Set up an **IAM Role** with access to **S3 & MongoDB**.  
4. Add the **MongoDB JDBC Connection** in AWS Glue.  

### **Step 2: Upload Dependencies**
- If using MongoDB Spark Connector, upload the **MongoDB Connector JAR** in **AWS Glue Libraries**.

### **Step 3: Schedule the Job**
- Run on a **schedule (e.g., every 10 min, hourly, or daily)** via **AWS Glue Triggers**.  

---

## **🔹 Next Steps**
1️⃣ **Optimize Queries**: Index `last_modified` in MongoDB.  
2️⃣ **Integrate with Athena**: Query S3 data using AWS Glue Catalog.  
3️⃣ **Build Analytics Pipelines**: Use **AWS EMR / AWS Glue ETL** for further transformations.  

🚀 **Would you like a version with Kafka for real-time streaming?**