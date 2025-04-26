If you're setting up a **new MacBook** for **Apache Airflow** development and contributing to open-source projects, you'll need a combination of system utilities, package managers, Python tools, and development environments. Here’s a step-by-step guide:

---

### **1. System Setup and Package Managers**
#### **Install Homebrew (MacOS Package Manager)**
Homebrew makes it easy to install software on macOS.
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
After installation, run:
```bash
brew update && brew upgrade
```

---

### **2. Install Python and Virtual Environment Tools**
Airflow requires **Python 3.8+**.
#### **Install Python 3.x**
```bash
brew install python
```
Verify installation:
```bash
python3 --version
```
#### **Set Up a Virtual Environment**
```bash
python3 -m venv airflow-env
source airflow-env/bin/activate
```

---

### **3. Install Apache Airflow**
Install Airflow in the virtual environment:
```bash
pip install --upgrade pip
pip install apache-airflow
```
To install Airflow with extra dependencies (e.g., Postgres, MySQL, Kubernetes):
```bash
pip install apache-airflow[postgres,mysql,kubernetes]
```

---

### **4. Install Database (PostgreSQL)**
Airflow needs a database (default: SQLite, but **PostgreSQL** is recommended).
#### **Install PostgreSQL**
```bash
brew install postgresql
brew services start postgresql
```
#### **Create an Airflow Database**
```bash
psql postgres
CREATE DATABASE airflow;
CREATE USER airflow WITH PASSWORD 'airflow';
ALTER ROLE airflow SET client_encoding TO 'utf8';
ALTER ROLE airflow SET default_transaction_isolation TO 'read committed';
ALTER ROLE airflow SET timezone TO 'UTC';
GRANT ALL PRIVILEGES ON DATABASE airflow TO airflow;
```
Set Airflow to use **PostgreSQL** in `airflow.cfg`:
```
sql_alchemy_conn = postgresql+psycopg2://airflow:airflow@localhost/airflow
```

---

### **5. Install Additional Dependencies**
#### **1. Install Airflow CLI**
```bash
pip install apache-airflow-cli
```
#### **2. Install Database Client Tools**
```bash
brew install mysql postgresql
```
#### **3. Install Docker (For Local Development)**
Docker is useful for testing Airflow with multiple services.
```bash
brew install --cask docker
```
#### **4. Install Apache Airflow Provider Packages**
To use Airflow with Google Cloud, AWS, and Kubernetes:
```bash
pip install apache-airflow[google,amazon,kubernetes]
```
#### **5. Install Celery for Task Queues**
```bash
pip install apache-airflow[celery]
brew install redis
```
Start Redis:
```bash
brew services start redis
```

---

### **6. Install Development Tools**
#### **1. Install Git**
```bash
brew install git
```
Set up Git:
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```
#### **2. Install Code Editor (VS Code)**
```bash
brew install --cask visual-studio-code
```
#### **3. Install PyCharm (Optional)**
```bash
brew install --cask pycharm
```

---

### **7. Install Kubernetes (Optional)**
If you want to run **Airflow on Kubernetes**, install:
```bash
brew install kubectl minikube
```
Start Minikube:
```bash
minikube start
```

---

### **8. Run Apache Airflow**
#### **1. Initialize the Database**
```bash
airflow db init
```
#### **2. Start Airflow Webserver**
```bash
airflow webserver --port 8080
```
#### **3. Start Airflow Scheduler**
```bash
airflow scheduler
```
#### **4. Open Airflow UI**
Open [http://localhost:8080](http://localhost:8080) in your browser.

---

### **9. Install Additional Tools for Open Source Contributions**
- **Pre-commit Hooks for Code Quality**
  ```bash
  pip install pre-commit
  pre-commit install
  ```
- **Linters and Formatters**
  ```bash
  pip install flake8 black isort
  ```
- **Test Frameworks**
  ```bash
  pip install pytest pytest-mock
  ```

---

### **Final Summary**
| **Category** | **Software** | **Installation Command** |
|-------------|-------------|---------------------------|
| **Package Manager** | Homebrew | `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"` |
| **Python & Virtual Environment** | Python 3, venv | `brew install python && python3 -m venv airflow-env` |
| **Apache Airflow** | Core Airflow | `pip install apache-airflow` |
| **Database** | PostgreSQL, MySQL | `brew install postgresql mysql` |
| **Development Tools** | Git, VS Code, PyCharm | `brew install git && brew install --cask visual-studio-code pycharm` |
| **Containerization** | Docker | `brew install --cask docker` |
| **Kubernetes (Optional)** | kubectl, Minikube | `brew install kubectl minikube` |
| **Linters & Testing** | Black, Flake8, Pytest | `pip install black flake8 pytest` |

---

### **Next Steps**
✅ **Set up Airflow DAGs**  
✅ **Contribute to Open Source (GitHub, GitLab)**  
✅ **Test locally using Docker/Kubernetes**  

Would you like help setting up an Airflow DAG or debugging a specific issue?