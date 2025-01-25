### **Developing an ETL Job Using AWS Glue and EMR**

In this workflow, AWS Glue handles the ETL (Extract, Transform, Load) orchestration, while Amazon EMR provides the computing infrastructure for running distributed Spark jobs. Here's how to set up and execute an ETL pipeline that combines both services.

---

### **1. Use Case**
- Extract data from an S3 bucket.
- Transform it using Spark (run on EMR).
- Load the transformed data back into an S3 bucket or a data warehouse like Amazon Redshift.

---

### **Step-by-Step Workflow**

#### **1. Setup Prerequisites**
- **AWS Glue**:
  - Create an AWS Glue Data Catalog for your source and target data.
  - Define databases and tables in the Glue Catalog.

Here’s how to create an **AWS Glue Data Catalog**, databases, and tables programmatically using **AWS CLI** and **Boto3 (Python SDK)**.

---

## **Using AWS CLI**

### 1. **Create a Glue Database**
Run the following AWS CLI command to create a database in the Glue Data Catalog:

```bash
aws glue create-database \
    --database-input '{"Name": "my_database", "Description": "My Glue database for ETL"}'
```

- Replace `my_database` with the desired database name.
- Optionally, add a `LocationUri` to specify where the data resides (e.g., an S3 bucket).

---

### 2. **Create a Glue Table**
Run the following AWS CLI command to create a table in the Glue Data Catalog:

```bash
aws glue create-table \
    --database-name my_database \
    --table-input '{
        "Name": "my_table",
        "Description": "Sample Glue Table",
        "StorageDescriptor": {
            "Columns": [
                {"Name": "id", "Type": "int"},
                {"Name": "name", "Type": "string"},
                {"Name": "age", "Type": "int"}
            ],
            "Location": "s3://my-bucket/my-folder/",
            "InputFormat": "org.apache.hadoop.mapred.TextInputFormat",
            "OutputFormat": "org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat",
            "SerdeInfo": {
                "SerializationLibrary": "org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe",
                "Parameters": {"field.delim": ","}
            }
        },
        "TableType": "EXTERNAL_TABLE"
    }'
```

- Replace `my_database` and `my_table` with your database and table names.
- Update the **`Location`** to point to your data in S3.

---

## **Using Python (Boto3)**

### **Setup**
Install the AWS SDK for Python (Boto3) if not already installed:

```bash
pip install boto3
```

### **Code to Create Glue Data Catalog, Database, and Table**

```python
import boto3

# Initialize the Glue client
glue = boto3.client('glue', region_name='us-east-1')  # Replace with your region

# Create a Glue Database
def create_database():
    response = glue.create_database(
        DatabaseInput={
            'Name': 'my_database',
            'Description': 'My Glue database for ETL'
        }
    )
    print("Database created:", response)

# Create a Glue Table
def create_table():
    response = glue.create_table(
        DatabaseName='my_database',
        TableInput={
            'Name': 'my_table',
            'Description': 'Sample Glue Table',
            'StorageDescriptor': {
                'Columns': [
                    {'Name': 'id', 'Type': 'int'},
                    {'Name': 'name', 'Type': 'string'},
                    {'Name': 'age', 'Type': 'int'}
                ],
                'Location': 's3://my-bucket/my-folder/',
                'InputFormat': 'org.apache.hadoop.mapred.TextInputFormat',
                'OutputFormat': 'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat',
                'SerdeInfo': {
                    'SerializationLibrary': 'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe',
                    'Parameters': {'field.delim': ','}
                }
            },
            'TableType': 'EXTERNAL_TABLE'
        }
    )
    print("Table created:", response)

# Run the functions
create_database()
create_table()
```

---

### **Explanation of Key Parameters**
- **Database Name**: Name of the database to be created.
- **Table Name**: Name of the table within the database.
- **Columns**: Schema of the table (name and type of each column).
- **Location**: Path to the data stored in S3.
- **Input/Output Format**: Specifies the file format for reading/writing data.
- **SerdeInfo**: Serialization and deserialization properties (e.g., delimiter for CSV).

---

### **Verify the Created Catalog**
1. Log in to the **AWS Management Console**.
2. Navigate to **AWS Glue** > **Data Catalog** > **Databases**.
3. Check if the database and table are listed.

---

Would you like help customizing this code for a specific use case?


- **Amazon EMR**:
  - Launch an EMR cluster with Spark installed.
  - Use an S3 bucket for EMR logs and intermediate data storage.

---

#### **2. Configure the AWS Glue ETL Job**
AWS Glue will orchestrate the job, connect to data sources, and submit the Spark job to the EMR cluster.

##### **Code Example for AWS Glue Script**

Below is a Python script for an AWS Glue ETL job using PySpark:

```python
from awsglue.context import GlueContext
from pyspark.context import SparkContext
from awsglue.dynamicframe import DynamicFrame

# Initialize GlueContext
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

# Configuration for AWS Glue Data Catalog
source_database = "source_db"
source_table = "source_table"
target_s3_path = "s3://my-target-bucket/transformed-data/"

# Extract data from Glue Data Catalog (S3 source)
source_dynamic_frame = glueContext.create_dynamic_frame.from_catalog(
    database=source_database,
    table_name=source_table
)

# Transform the data (example transformation: filtering rows)
transformed_dynamic_frame = source_dynamic_frame.filter(
    lambda row: row["age"] and int(row["age"]) > 30
)

# Convert DynamicFrame to DataFrame for Spark transformations (optional)
transformed_df = transformed_dynamic_frame.toDF()

# Perform additional transformations with Spark DataFrame (optional)
transformed_df = transformed_df.withColumnRenamed("name", "full_name")

# Convert DataFrame back to DynamicFrame
final_dynamic_frame = DynamicFrame.fromDF(transformed_df, glueContext, "final_dynamic_frame")

# Load transformed data back to S3
glueContext.write_dynamic_frame.from_options(
    frame=final_dynamic_frame,
    connection_type="s3",
    connection_options={"path": target_s3_path},
    format="parquet"
)
```

---

#### **3. Submit the Spark Job to EMR**
You can configure AWS Glue to submit the Spark job to an EMR cluster by using an **EMR endpoint**. Here's how:

- Specify the **EMR cluster's master node** as the endpoint for your Spark jobs.
- Enable integration with AWS Glue.

---

#### **4. Launch an EMR Cluster**
You can launch an EMR cluster with Spark installed using the AWS Management Console or AWS CLI.

##### **AWS CLI Example**
```bash
aws emr create-cluster \
    --name "ETL Cluster" \
    --release-label emr-6.5.0 \
    --applications Name=Spark Name=Hadoop \
    --instance-type m5.xlarge \
    --instance-count 3 \
    --use-default-roles \
    --log-uri s3://my-log-bucket/emr-logs/ \
    --auto-terminate
```

---

#### **5. Glue Configuration to Use EMR**
In the AWS Glue job configuration, specify the following:
1. Select the **“Job Parameters”** tab and set `--enable-spark-submit` to **true**.
2. Use the EMR endpoint as the target for submitting the Spark job.

---

#### **6. Run the Job**
- Trigger the Glue ETL job from the AWS Management Console, AWS CLI, or programmatically using the AWS SDK.
- Monitor the job status in AWS Glue and EMR (logs can be found in Amazon CloudWatch or the specified S3 log path).

To create an **AWS Glue job that submits a Spark job to an EMR cluster** using the AWS CLI, you need to configure the Glue job to use the EMR cluster as the Spark execution environment. Below are the steps and a detailed example to achieve this.

---

### **1. Pre-requisites**
1. **Create an EMR Cluster**:
   - Launch an EMR cluster with Spark installed. You can use the following AWS CLI command:
   ```bash
   aws emr create-cluster \
       --name "ETL Cluster" \
       --release-label emr-6.5.0 \
       --applications Name=Spark Name=Hadoop \
       --instance-type m5.xlarge \
       --instance-count 3 \
       --use-default-roles \
       --log-uri s3://my-log-bucket/emr-logs/ \
       --auto-terminate
   ```

   Take note of the **Cluster ID** and **Master Node DNS** for later use.

2. **Prepare the Spark Script**:
   - Save your Spark job script (e.g., `etl_script.py`) to an S3 bucket, as Glue jobs require access to this script.

3. **IAM Roles**:
   - Ensure you have an **IAM role** with sufficient permissions to interact with Glue, EMR, and S3.

---

### **2. Create the Glue Job Using AWS CLI**
Here’s how to create a Glue job to submit your Spark script to the EMR cluster.

#### **Command to Create the Glue Job**
```bash
aws glue create-job \
    --name my-emr-glue-job \
    --role AWSGlueServiceRole \
    --command '{"Name": "glueetl", "ScriptLocation": "s3://my-scripts-bucket/etl_script.py"}' \
    --default-run-properties '{"--job-bookmark-option": "job-bookmark-enable"}' \
    --connections Connections="MyEMRClusterConnection" \
    --glue-version "2.0" \
    --number-of-workers 5 \
    --worker-type Standard \
    --execution-class FLEX \
    --description "Glue job to submit Spark job to EMR"
```

---

### **Explanation of Key Parameters**
- **`--name`**: The name of your Glue job.
- **`--role`**: The IAM role that Glue will use to access EMR, S3, and other resources.
- **`--command`**: Specifies the Spark script location in S3 and the type of job (`glueetl`).
  - `ScriptLocation`: The path to the Spark script in S3.
- **`--connections`**: Use an **EMR connection** to link the Glue job to the EMR cluster.
- **`--glue-version`**: Glue version (2.0 or 3.0) that determines the underlying Spark and Python versions.
- **`--number-of-workers`**: Number of Glue workers for the job.
- **`--worker-type`**: Type of workers (`Standard`, `G.1X`, or `G.2X`).
- **`--execution-class`**: `FLEX` for cost-optimized execution or `STANDARD` for low-latency.

---

### **3. Create the EMR Connection**
You need to create a Glue connection to your EMR cluster. This allows Glue to communicate with the EMR master node.

#### **Command to Create Connection**
```bash
aws glue create-connection \
    --name MyEMRClusterConnection \
    --connection-input '{"Name": "MyEMRClusterConnection", 
                         "ConnectionType": "NETWORK", 
                         "ConnectionProperties": { 
                             "HOST": "emr-master-node-dns", 
                             "PORT": "8998", 
                             "USERNAME": "emr-user", 
                             "PASSWORD": "emr-password" 
                         }
                        }'
```

- Replace **`emr-master-node-dns`** with the DNS name of your EMR master node.
- Replace **`emr-user`** and **`emr-password`** with appropriate credentials for the EMR cluster.

---

### **4. Start the Glue Job**
Once the Glue job is created, you can run it using the following AWS CLI command:

#### **Command to Start the Glue Job**
```bash
aws glue start-job-run \
    --job-name my-emr-glue-job
```

---

### **5. Monitor the Glue Job**
To check the status of the Glue job, use:

#### **Command to Check Job Status**
```bash
aws glue get-job-run \
    --job-name my-emr-glue-job \
    --run-id <job-run-id>
```

Replace `<job-run-id>` with the Job Run ID returned when you started the job.

---

### **6. Logs and Debugging**
- **Glue Logs**: Available in Amazon CloudWatch under the Glue job's log group.
- **EMR Logs**: Available in the S3 bucket specified during EMR cluster creation (e.g., `s3://my-log-bucket/emr-logs/`).

---

### **7. Sample Spark Script (etl_script.py)**

Here’s a basic Spark script to process data from S3 and save the output:

```python
from pyspark.sql import SparkSession

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("EMR Glue Job Example") \
    .getOrCreate()

# Read data from S3
input_path = "s3://my-input-bucket/raw-data/"
output_path = "s3://my-output-bucket/processed-data/"

df = spark.read.csv(input_path, header=True, inferSchema=True)

# Perform some transformations
df_transformed = df.filter(df['age'] > 30)

# Write the transformed data back to S3
df_transformed.write.parquet(output_path, mode="overwrite")

spark.stop()
```

---

### **Advantages of Using AWS Glue with EMR**
1. **Scalability**: EMR handles large-scale Spark workloads, while Glue manages orchestration.
2. **Cost-Effective**: Use Glue’s serverless nature and EMR’s auto-scaling for optimized cost.
3. **Ease of Management**: Glue simplifies job scheduling and integrates seamlessly with other AWS services.
4. **Flexible Execution**: Leverage Glue's integration with an external Spark runtime (EMR).

---

Would you like more details on a specific step or customization for your use case?

---

### **Architecture Diagram**
1. **Extract**: AWS Glue reads data from the Glue Data Catalog (S3 source).
2. **Transform**: Spark transformations are executed on an EMR cluster.
3. **Load**: The transformed data is written back to S3 or another destination like Redshift.

```
[S3 Source] --> [Glue ETL Job] --> [EMR Spark Job] --> [S3/Redshift]
```

---

### **Advantages of Combining Glue and EMR**
1. **Flexibility**: Use Glue for orchestration and EMR for heavy compute workloads.
2. **Scalability**: EMR provides scalable Spark clusters for processing large datasets.
3. **AWS Integration**: Seamless integration with S3, Redshift, and Glue Data Catalog.
4. **Cost-Effectiveness**: EMR’s auto-scaling and Glue’s serverless nature reduce idle costs.

---

Would you like help setting up any specific part of this workflow?