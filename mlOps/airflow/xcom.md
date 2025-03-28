### **What is XCom in Apache Airflow?**  
**XCom (Cross-Communication)** in Apache Airflow is a mechanism that allows tasks to share information with each other. It enables **data passing** between tasks within a DAG (Directed Acyclic Graph).

---

## **How XCom Works?**
- Tasks in Airflow **push** values to XCom, and other tasks can **pull** those values when needed.
- The data is stored in **Airflow’s metadata database**.
- XComs are key-value pairs with **metadata** (e.g., timestamp, execution date).

---

## **Key XCom Functions in Airflow**
### **1. `xcom_push` (Store Data)**
Used to push data from one task.
```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def push_data(**kwargs):
    kwargs['ti'].xcom_push(key='message', value='Hello from Task 1')

with DAG('xcom_example', start_date=datetime(2024, 1, 1), schedule_interval=None) as dag:
    task1 = PythonOperator(
        task_id='push_task',
        python_callable=push_data,
        provide_context=True
    )
```
- Stores `'Hello from Task 1'` in XCom with key `'message'`.

---

### **2. `xcom_pull` (Retrieve Data)**
Used to pull data in another task.
```python
def pull_data(**kwargs):
    message = kwargs['ti'].xcom_pull(task_ids='push_task', key='message')
    print(f"Received Message: {message}")

task2 = PythonOperator(
    task_id='pull_task',
    python_callable=pull_data,
    provide_context=True
)
```
- Retrieves the value stored by `push_task` using the **task ID** and **key**.

---

## **XCom Storage and UI**
- XComs are stored in the **`xcom` table** in Airflow’s metadata database.
- You can view them in the **Airflow UI** under `Admin -> XComs`.

---

## **Limitations of XCom**
- **Size Limit:** XCom stores data in the database, so large payloads (e.g., big files) should not be passed via XCom. Use **S3, GCS, or databases** instead.
- **Serialization:** XComs store values as **pickled objects**, so they must be serializable.

---

## **Alternative to XCom for Large Data**
If your data is large, store it in an **external system** like:
- **Amazon S3 / Google Cloud Storage** (GCS)
- **PostgreSQL / MySQL**
- **Redis / Kafka**
- Then pass the reference (file path or database ID) using XCom.

---

## **Conclusion**
- **XCom is useful** for sharing small data between tasks.
- Use **`xcom_push()`** and **`xcom_pull()`** for communication.
- Avoid storing large data—use **external storage** instead.

Would you like an example of XCom using the `@task` decorator in **Airflow 2+**?


### **XCom in Airflow 2+ using the `@task` Decorator**  
Apache Airflow 2+ introduced the `@task` decorator, which makes task definition **simpler and more readable**. It also **automatically pushes return values** to XCom.

---

## **Example: Passing Data Between Tasks Using `@task` Decorator**
```python
from airflow.decorators import dag, task
from datetime import datetime

# Define the DAG
@dag(schedule=None, start_date=datetime(2024, 1, 1), catchup=False)
def xcom_example_dag():
    
    # Task 1: Push data (implicitly returns a value to XCom)
    @task
    def push_task():
        return "Hello from Task 1"

    # Task 2: Pull data from XCom (implicit `xcom_pull`)
    @task
    def pull_task(message):
        print(f"Received Message: {message}")

    # Define task execution order
    message = push_task()  # push_task() automatically pushes return value to XCom
    pull_task(message)  # pull_task receives it automatically

# Instantiate the DAG
xcom_example_dag()
```

---

## **How This Works**
1. **`push_task()` returns a value**, which is automatically stored in XCom.
2. **`pull_task(message)`** automatically pulls the value from XCom **without explicitly calling `xcom_pull()`**.
3. The **data flow** happens **implicitly** when tasks are assigned (`message = push_task()`).

---

## **Advantages of Using `@task` Decorator**
✅ **No Need for `xcom_push()` / `xcom_pull()`** – Return values are **automatically stored in XCom**.  
✅ **More Readable Code** – Task dependencies are defined **inline**.  
✅ **Better for Small Data** – Ideal for passing small JSON-like objects.  

Would you like an example where **XCom is used with external storage** (e.g., S3, Redis)?

### **Using XCom with External Storage (Amazon S3) in Airflow 2+**
Since **XCom stores data in the database**, it is not ideal for large files or big data. Instead, we can **store data in Amazon S3** (or another storage system) and pass the reference (file path or object key) via XCom.

---

## **Example: Storing and Retrieving Data from S3 Using XCom**
### **Prerequisites:**
1. **Install `boto3`** (AWS SDK for Python):
   ```bash
   pip install boto3
   ```
2. **Ensure Airflow is configured with AWS credentials** (IAM role or `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` in environment variables).

---

### **DAG: Store Data in S3 and Pass S3 Path via XCom**
```python
import boto3
from airflow.decorators import dag, task
from datetime import datetime
import json

# Define S3 bucket and file name
S3_BUCKET = "my-airflow-bucket"
S3_FILE_KEY = "data/employee.json"

# Initialize S3 Client
s3_client = boto3.client("s3")

@dag(schedule=None, start_date=datetime(2024, 1, 1), catchup=False)
def xcom_s3_example():

    @task
    def upload_to_s3():
        # Sample data to be stored in S3
        data = {"id": 1, "name": "John Doe", "role": "Engineer"}
        json_data = json.dumps(data)

        # Upload data to S3
        s3_client.put_object(Bucket=S3_BUCKET, Key=S3_FILE_KEY, Body=json_data)
        print(f"Uploaded data to S3: {S3_FILE_KEY}")

        # Return the file path as an XCom value
        return S3_FILE_KEY  

    @task
    def fetch_from_s3(file_key):
        # Download data from S3
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=file_key)
        data = json.loads(response["Body"].read().decode("utf-8"))

        print(f"Fetched Data from S3: {data}")
        return data  # Can be used by another task if needed

    # Task execution order
    file_path = upload_to_s3()  # Stores the file key in XCom
    fetch_from_s3(file_path)  # Retrieves it from XCom

# Instantiate the DAG
xcom_s3_example()
```

---

## **How This Works**
1. **`upload_to_s3()`**
   - Uploads JSON data to **Amazon S3**.
   - Stores the **S3 object key** (file path) in XCom.

2. **`fetch_from_s3(file_key)`**
   - **Retrieves the S3 file key from XCom**.
   - Downloads the JSON data from S3 and prints it.

---

## **Advantages of This Approach**
✅ **Efficient for Large Data** – Avoids storing large payloads in XCom.  
✅ **Scalability** – S3 is built for **high throughput** and **storage efficiency**.  
✅ **Works with Any External Storage** – Can be adapted for **Google Cloud Storage (GCS), Redis, PostgreSQL, etc.**  

Would you like an example of **XCom with a database (e.g., PostgreSQL)?**
