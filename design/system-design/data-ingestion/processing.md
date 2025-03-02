### **Enhanced Data Processing Pipeline: Raw to Processed Bucket**

Your updated system will now:
1. **Ingest Data**: IoT events are stored in the **S3 raw bucket**.
2. **Process Data**: Extract, transform, and load (ETL) using **AWS Glue** and/or **Spring Boot microservices**.
3. **Store in Processed Bucket**: Move structured data to the **S3 processed bucket**.
4. **Use AWS EFS if Required**: For temporary storage or microservices needing shared access.

---

## **1. AWS Glue for ETL Processing**
AWS Glue can **process raw JSON files** and store the cleaned data in **S3 processed bucket**.

### **Step 1: AWS Glue Job (Python)**
Create an AWS Glue job (`glue_etl.py`):
```python
import sys
import json
import boto3
from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import SparkSession

# Initialize Spark
spark = SparkSession.builder.appName("IoTDataProcessing").getOrCreate()
glueContext = GlueContext(spark)
s3 = boto3.client('s3')

# Read from Raw Bucket
raw_bucket = "your-raw-bucket"
processed_bucket = "your-processed-bucket"

df = spark.read.json(f"s3://{raw_bucket}/access_logs/*.json")

# Transform Data
df = df.withColumnRenamed("device_id", "DeviceID").withColumnRenamed("user_id", "UserID")

# Write to Processed Bucket
df.write.mode("overwrite").json(f"s3://{processed_bucket}/processed_logs/")

print("Data processing complete, moved to processed bucket")
```

### **Step 2: Deploy AWS Glue Job**
- Upload the script to **S3**.
- Create an **AWS Glue Job**.
- Set the job to **run every 5 minutes** using **AWS Glue Scheduler**.

---

## **2. Spring Boot Microservice for On-Demand Processing**
For more **control over processing**, use a **Spring Boot microservice**.

### **Step 1: Create Microservice (`S3ProcessorService`)**
```java
@RestController
@RequestMapping("/process")
public class S3ProcessorController {

    @Autowired
    private S3ProcessingService processingService;

    @PostMapping("/trigger")
    public ResponseEntity<String> triggerProcessing() {
        processingService.processRawData();
        return ResponseEntity.ok("Processing Started");
    }
}
```

### **Step 2: Read & Process Data**
```java
@Service
public class S3ProcessingService {

    private final AmazonS3 s3Client;

    public S3ProcessingService() {
        this.s3Client = AmazonS3ClientBuilder.standard().withRegion(Regions.US_EAST_1).build();
    }

    public void processRawData() {
        ListObjectsV2Result result = s3Client.listObjectsV2("your-raw-bucket", "access_logs/");
        for (S3ObjectSummary file : result.getObjectSummaries()) {
            S3Object s3Object = s3Client.getObject("your-raw-bucket", file.getKey());
            // Process data (Transform, Clean)
            s3Client.putObject("your-processed-bucket", "processed_logs/" + file.getKey(), transformedData);
        }
    }
}
```
✅ Exposes `/process/trigger` API to **manually trigger processing**.  
✅ Moves cleaned data to **S3 processed bucket**.

---

## **3. AWS EFS for Shared Storage**
If microservices **need shared storage**:
- Mount **AWS EFS** on Kubernetes pods for persistent storage.
- Store intermediate processed data.

### **Mounting EFS in Kubernetes**
```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: efs-pv
spec:
  capacity:
    storage: 10Gi
  accessModes:
    - ReadWriteMany
  persistentVolumeReclaimPolicy: Retain
  csi:
    driver: efs.csi.aws.com
    volumeHandle: "fs-12345678"
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: efs-claim
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 5Gi
```
✅ **Used by microservices** for temporary storage.

---

## **4. Summary**
- **AWS Glue** → Scheduled ETL processing (JSON → Cleaned Data).
- **Spring Boot** → Manual/On-demand processing.
- **AWS EFS** → Shared storage for intermediate processing.

Would you like help **deploying this in AWS EKS**?