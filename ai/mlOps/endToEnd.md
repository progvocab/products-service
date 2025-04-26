### **End-to-End ML Pipeline: Combining Airflow, Kubeflow, and MLflow**  

To build a **scalable and production-ready ML pipeline**, we can integrate:  
- **Airflow** → Orchestrates the workflow, schedules tasks (data ingestion, training, deployment).  
- **Kubeflow** → Manages ML training, hyperparameter tuning, and model serving on Kubernetes.  
- **MLflow** → Tracks experiments, logs model performance, and manages deployments.  

---

## **1️⃣ Architecture Overview**  
Here’s how these tools work together:  

1. **Airflow triggers the pipeline** → Extracts data, preprocesses, and sends it to Kubeflow.  
2. **Kubeflow runs ML tasks** → Training, hyperparameter tuning, and serving the model.  
3. **MLflow tracks experiments** → Logs parameters, metrics, and model artifacts.  
4. **Airflow deploys the best model** → Retrieves the best model from MLflow and deploys it to an API.  

🚀 **Stack Used:**  
- **Airflow** (Scheduler & Workflow Orchestration)  
- **Kubeflow Pipelines** (ML Model Training & Serving on Kubernetes)  
- **MLflow** (Experiment Tracking & Model Management)  
- **MinIO/S3/GCS** (Model Artifact Storage)  
- **FastAPI/Flask** (Model Serving)  

---

## **2️⃣ Workflow Breakdown**
| **Step** | **Tool Used** | **Description** |
|----------|-------------|----------------|
| **1. Data Ingestion** | **Airflow** | Pulls data from a database, API, or storage (S3, GCS, etc.). |
| **2. Preprocessing** | **Airflow/Kubeflow** | Cleans, transforms, and prepares data for training. |
| **3. Model Training** | **Kubeflow** | Trains multiple models and tunes hyperparameters. |
| **4. Experiment Tracking** | **MLflow** | Logs model performance, hyperparameters, and results. |
| **5. Model Selection** | **Airflow + MLflow** | Retrieves the best model based on metrics. |
| **6. Deployment** | **Airflow + Kubernetes** | Deploys the selected model as an API (FastAPI/Flask). |
| **7. Monitoring & Retraining** | **Airflow + MLflow** | Monitors drift and triggers retraining when needed. |

---

## **3️⃣ Implementation: Code for Each Step**
### **📌 Step 1: Airflow DAG - Triggering the ML Pipeline**
**Airflow DAG to orchestrate ML pipeline execution.**
```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import subprocess

def trigger_kubeflow_pipeline():
    subprocess.run(["python", "kubeflow_pipeline.py"])  # Calls Kubeflow pipeline script

default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 2, 1),
    "catchup": False,
}

with DAG("ml_pipeline", default_args=default_args, schedule_interval="@daily") as dag:
    start_pipeline = PythonOperator(
        task_id="start_ml_pipeline",
        python_callable=trigger_kubeflow_pipeline,
    )

    start_pipeline
```
---

### **📌 Step 2: Kubeflow Pipeline - Model Training & Hyperparameter Tuning**
**Kubeflow Pipeline to train an ML model.**
```python
import kfp
from kfp.dsl import pipeline, component
import mlflow

@component
def train_model():
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    
    X = np.random.rand(100, 1) * 10
    y = 2 * X + np.random.randn(100, 1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = LinearRegression()
    model.fit(X_train, y_train)

    mlflow.log_param("coef", model.coef_[0][0])
    mlflow.log_metric("score", model.score(X_test, y_test))
    mlflow.sklearn.log_model(model, "linear_regression_model")

    return mlflow.active_run().info.run_id

@pipeline(name="ML Pipeline")
def ml_pipeline():
    train_task = train_model()

if __name__ == "__main__":
    from kfp.compiler import Compiler
    Compiler().compile(ml_pipeline, "ml_pipeline.yaml")
```
🚀 This pipeline:
- **Trains a model**
- **Logs parameters & metrics in MLflow**
- **Saves the best model to MLflow**

---

### **📌 Step 3: MLflow - Model Experiment Tracking**
**MLflow Logs the best model and saves it.**
```python
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
import numpy as np

mlflow.set_experiment("ml_training_experiment")

X = np.random.rand(100, 1) * 10
y = 2 * X + np.random.randn(100, 1)

with mlflow.start_run():
    model = LinearRegression()
    model.fit(X, y)

    mlflow.log_param("coef", model.coef_[0][0])
    mlflow.log_metric("score", model.score(X, y))
    
    mlflow.sklearn.log_model(model, "linear_regression_model")
```
---

### **📌 Step 4: Airflow - Deploying the Best Model**
Airflow fetches the best model from MLflow and deploys it.
```python
import mlflow
from mlflow.tracking import MlflowClient
import shutil

def fetch_best_model():
    client = MlflowClient()
    best_run = max(client.search_runs(experiment_ids=["1"]), key=lambda run: run.data.metrics["score"])
    model_path = best_run.info.artifact_uri + "/linear_regression_model"
    
    shutil.copytree(model_path.replace("file://", ""), "/deployment/model")  # Copy to deployment directory

fetch_best_model()
```
---

### **📌 Step 5: Deploy Model as a REST API (FastAPI)**
FastAPI serves the best ML model.
```python
from fastapi import FastAPI
import mlflow.pyfunc

app = FastAPI()
model = mlflow.pyfunc.load_model("/deployment/model")

@app.post("/predict/")
def predict(data: float):
    return {"prediction": model.predict([[data]])[0]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```
---

## **4️⃣ Deployment & Execution**
### **Deploying This Pipeline**
1. **Start Airflow**:  
   ```bash
   airflow scheduler & airflow webserver
   ```
2. **Deploy Kubeflow Pipelines**:  
   ```bash
   kubectl apply -f ml_pipeline.yaml
   ```
3. **Run MLflow Server**:  
   ```bash
   mlflow server --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000
   ```
4. **Serve the Model API**:  
   ```bash
   python serve_model.py
   ```

---

## **5️⃣ Summary**
- **Airflow** orchestrates the entire ML workflow.
- **Kubeflow** trains and serves ML models efficiently.
- **MLflow** tracks experiments and manages models.
- **FastAPI** serves the best model as an API.

---

## **6️⃣ When to Use This Setup?**
| **Use Case** | **Recommended Approach** |
|--------------|-------------------------|
| **ETL & ML Pipelines** | Airflow + Kubeflow |
| **Scalable ML Training** | Kubeflow |
| **Experiment Tracking & Model Registry** | MLflow |
| **Automated Model Deployment** | Airflow + MLflow |
| **Kubernetes-based ML workflows** | Kubeflow |

Would you like more details on **deployment, monitoring, or scaling?** 🚀
