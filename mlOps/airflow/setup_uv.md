
## Prerequisites

* Install uv 


## 1. Create Virtual Environment

```
cd git\airflow
uv venv
```
## 2. Start virtual environment
```
source .venv/bin/activate
```

### run the following within virtual environment
```
uv pip install -e ".[devel,google]" \
  --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-main/constraints-source-providers-3.9.txt"
```
```
uv sync --extra devel --extra devel-tests --extra google
```


### DB setup
```
airflow db migrate
```
* In case of errors

```
airflow db reset
```

### Start Airflow Scheduler 
```
airflow scheduler
```

### Start Airflow Webserver
```
  airflow webserver --port 8080
```

### Create User 
 
Run the following command to create a user:

```sh
airflow users create \
    --username admin \
    --password admin \
    --firstname John \
    --lastname Doe \
    --role Admin \
    --email admin@example.com
```

### Run test

* All tests
```
pytest --with-db-init
```
* Single File 


## 3. Start Docker

```
colima start
```
## 4. Start kubernates
minikube start
### **Creating a User with a Password in Apache Airflow from the Terminal**  

Apache Airflow provides a CLI command to create a user with a password. Follow these steps:

---

### **1. Ensure Airflow is Installed and Initialized**
If you haven’t initialized Airflow yet, do this first:
```sh
airflow db init
```

---


📌 **Parameters Explanation:**
- `--username` → The login username  
- `--password` → The password for the user  
- `--firstname` → First name of the user  
- `--lastname` → Last name of the user  
- `--role` → The role of the user (e.g., **Admin, User, Op, Viewer**)  
- `--email` → The email ID of the user  

---

### **3. Verify the User**
After creating the user, you can list all users to verify:
```sh
airflow users list
```
It will show something like:
```
+----+-----------+------------+-----------+---------------+
| id | username | first_name | last_name | role          |
+----+-----------+------------+-----------+---------------+
| 1  | admin    | John       | Doe       | Admin         |
+----+-----------+------------+-----------+---------------+
```

---

### **4. Start Airflow Webserver**
To use this user to log in, start the webserver:
```sh
airflow webserver --port 8080
```
Then go to **`http://localhost:8080`**, and log in with the credentials:

- **Username:** `admin`
- **Password:** `admin123`

---

### **5. (Optional) Delete a User**
If you need to remove a user:
```sh
airflow users delete --username admin
```

---

## **Summary**
| **Action** | **Command** |
|------------|------------|
| Create User | `airflow users create --username admin --password admin123 --firstname John --lastname Doe --role Admin --email admin@example.com` |
| List Users | `airflow users list` |
| Delete User | `airflow users delete --username admin` |

Would you like help with anything else related to Airflow? 🚀
