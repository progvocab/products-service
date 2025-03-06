### **What is Flake8 in Python?**
**Flake8** is a **Python linting tool** that checks your code for **style guide violations, logical errors, and complexity issues**. It is not a testing tool like `pytest`, but it helps maintain code quality by enforcing coding standards.  

---

### **Why Use Flake8?**
✅ Ensures your code follows **PEP 8 (Python's official style guide)**  
✅ Detects **unused imports, undefined variables, and syntax errors**  
✅ Identifies **complex functions (cyclomatic complexity)**  
✅ Helps maintain **readable and consistent code**  

---

### **How to Install Flake8**
You can install it via pip:  
```sh
pip install flake8
```

---

### **How to Use Flake8**
#### **1. Running Flake8 on a Python File**
```sh
flake8 my_script.py
```
If your code has issues, Flake8 outputs warnings like:
```
my_script.py:10:5: F841 local variable 'x' is assigned but never used
```

#### **2. Running Flake8 on a Project**
```sh
flake8 .
```
This checks all Python files in the current directory.

---

### **Flake8 in a CI/CD Pipeline (Jenkins)**
You can add a **Flake8 stage** to your **Jenkinsfile**:
```groovy
stage('Code Linting with Flake8') {
    steps {
        script {
            sh "flake8 . --max-line-length=120"
        }
    }
}
```
If Flake8 finds violations, it will fail the build.

---

### **Customizing Flake8 Rules**
You can configure Flake8 in a `.flake8` or `setup.cfg` file:
```ini
[flake8]
max-line-length = 120
exclude = venv, migrations
ignore = E203, W503
```

---

### **Flake8 vs Pytest**
| **Feature**  | **Flake8**  | **Pytest**  |
|-------------|------------|------------|
| Purpose  | Code linting & style checks | Unit & functional testing |
| Detects  | Style violations, complexity | Test failures, logic errors |
| Example Issue | "Line too long (121 > 120 chars)" | "AssertionError: expected 10, got 5" |

Would you like help integrating **Flake8 with Pytest** or **Jenkins reporting**?