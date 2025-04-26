### **Creating a New DAG in Apache Airflow** 🚀  

A **DAG (Directed Acyclic Graph)** in Apache Airflow is a workflow definition that consists of **tasks** and **dependencies**. Below are the steps to create and run a new DAG.

---

## **1️⃣ Create a DAG Directory**
Ensure your DAG files are stored in the correct directory. The default location is:

```sh
~/airflow/dags/
```
If you're using a different location, check with:
```sh
airflow config get-value core dags_folder
```

---

## **2️⃣ Create a New DAG File**
1. Open your DAGs folder:
```sh
cd ~/airflow/dags
```
2. Create a new Python DAG file:
```sh
nano my_first_dag.py
```

3. Add the following DAG code:

```python
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

# Define default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 2, 1),  # Adjust as needed
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Create the DAG
dag = DAG(
    'my_first_dag',
    default_args=default_args,
    description='A simple example DAG',
    schedule_interval=timedelta(days=1),  # Runs daily
    catchup=False
)

# Define a simple Python function
def print_hello():
    print("Hello, Apache Airflow!")

# Create a PythonOperator task
task1 = PythonOperator(
    task_id='print_hello',
    python_callable=print_hello,
    dag=dag,
)

# Define task dependencies (Optional in this case)
task1
```

📌 **What this DAG does:**  
- Runs **daily** (`schedule_interval=timedelta(days=1)`)  
- Starts on `2024-02-01` (`start_date`)  
- Prints **"Hello, Apache Airflow!"** when executed  

---

## **3️⃣ Restart Airflow to Detect the DAG**
Once the file is saved, restart Airflow to ensure the DAG is detected:

```sh
airflow scheduler &
airflow webserver --port 8080 &
```

---

## **4️⃣ Enable and Trigger the DAG**
1. Open the **Airflow Web UI** at [http://localhost:8080](http://localhost:8080).  
2. Navigate to the **"DAGs"** tab.  
3. Find **"my_first_dag"** in the list.  
4. Click the **toggle button** to enable it.  
5. Click the **Play (▶️) button** and choose **Trigger DAG**.  

---

## **5️⃣ Monitor DAG Execution**
You can monitor execution in multiple ways:

- **Web UI**: Go to the **Graph View** or **Task Instance Logs**.  
- **Logs from CLI**:
  ```sh
  airflow tasks logs my_first_dag print_hello
  ```

---

## **6️⃣ (Optional) List and Delete DAGs**
To **list all DAGs**:
```sh
airflow dags list
```

To **delete a DAG** (remove all records but keep the file):
```sh
airflow dags delete my_first_dag
```

To **remove the DAG completely**, delete the `.py` file:
```sh
rm ~/airflow/dags/my_first_dag.py
```

---

### **🔹 Summary**
| **Step** | **Command/Action** |
|----------|------------------|
| Create a new DAG file | `nano ~/airflow/dags/my_first_dag.py` |
| Restart Airflow | `airflow scheduler & airflow webserver --port 8080 &` |
| View DAGs | `airflow dags list` |
| Trigger DAG manually | `airflow dags trigger my_first_dag` |
| Monitor logs | `airflow tasks logs my_first_dag print_hello` |

Would you like help with more advanced DAGs, like dynamic tasks or dependencies? 🚀
