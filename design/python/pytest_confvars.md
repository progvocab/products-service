# **Understanding `conf_vars` and Other Important Pytest Decorators**  

Pytest provides a variety of **decorators** to modify test behavior, including `conf_vars`, which is commonly used in **Apache Airflow testing**.  

---

## **1. `@conf_vars` Decorator in Pytest (Used in Apache Airflow)**  

### **What is `conf_vars`?**  
`@conf_vars` is a **pytest decorator used in Apache Airflow** to **temporarily override configuration values** for a test. It modifies `airflow.configuration.conf` values **only during the test execution**.  

### **Example Usage in Airflow Tests**  
```python
from airflow.configuration import conf
from airflow.utils.test_utils import conf_vars

@conf_vars({("core", "load_examples"): "False"})
def test_airflow_config_override():
    assert conf.getboolean("core", "load_examples") is False
```
### **Explanation:**  
- **Before the test** → `conf["core"]["load_examples"]` could be `True`.  
- **During the test** → It is overridden to `False`.  
- **After the test** → The original value is restored.  

---

## **2. Other Important Pytest Decorators**  

### **2.1. `@pytest.mark.skip` → Skip a Test**  
Temporarily **disable a test**.  

```python
import pytest

@pytest.mark.skip(reason="This feature is not implemented yet")
def test_not_ready():
    assert 1 + 1 == 2
```
✅ Output: **Test is skipped**.

---

### **2.2. `@pytest.mark.skipif` → Conditional Skip**  
Skip a test **based on a condition**.  

```python
import pytest
import sys

@pytest.mark.skipif(sys.version_info < (3, 9), reason="Requires Python 3.9+")
def test_python_version():
    assert sys.version_info >= (3, 9)
```
✅ Runs only on **Python 3.9+**.

---

### **2.3. `@pytest.mark.xfail` → Expected Failure**  
Marks a test as **expected to fail** without marking it as an error.  

```python
import pytest

@pytest.mark.xfail(reason="Bug not fixed yet")
def test_fail_case():
    assert 1 / 0 == 0
```
✅ Output: Test **fails** but is reported as **expected**.

---

### **2.4. `@pytest.mark.parametrize` → Run a Test with Multiple Inputs**  
Executes the **same test multiple times** with different parameters.  

```python
import pytest

@pytest.mark.parametrize("x, y, expected", [(2, 3, 5), (1, 5, 6), (0, 0, 0)])
def test_addition(x, y, expected):
    assert x + y == expected
```
✅ Runs **three times** with different values.

---

### **2.5. `@pytest.fixture` → Reusable Test Setup**  
Defines a **setup function** to be used in multiple tests.  

```python
import pytest

@pytest.fixture
def sample_data():
    return {"name": "Alice", "age": 30}

def test_sample(sample_data):
    assert sample_data["name"] == "Alice"
    assert sample_data["age"] == 30
```
✅ The fixture **automatically provides test data**.

---

### **2.6. `@pytest.mark.usefixtures` → Use a Fixture Without Passing as Argument**  
Useful when **you don’t need to pass the fixture explicitly** in the test function.  

```python
import pytest

@pytest.fixture
def setup_env():
    print("Setting up environment")

@pytest.mark.usefixtures("setup_env")
def test_env():
    assert True
```
✅ Runs **setup code before the test**.

---

### **2.7. `@pytest.mark.timeout` → Set Time Limit for a Test**  
Fails the test if it **runs longer than a specified time**.  

```python
import pytest
import time

@pytest.mark.timeout(2)
def test_slow_function():
    time.sleep(1)  # Runs within limit
```
✅ If execution **exceeds 2 seconds**, the test **fails**.

---

### **2.8. `@pytest.mark.slow` → Custom Markers**  
You can create **custom markers** to categorize tests.  

```python
import pytest

@pytest.mark.slow
def test_large_data():
    assert sum(range(1000000)) > 0
```
✅ Run only slow tests:
```bash
pytest -m slow
```

---

### **3. Running Pytest with Decorators**  
Use the following commands:  
```bash
pytest -v  # Run all tests with verbose output
pytest -m slow  # Run only slow tests
pytest --runxfail  # Run tests expected to fail
```

---

## **4. Conclusion**  
- `@conf_vars` is **specific to Apache Airflow** for **overriding configs temporarily**.  
- Pytest provides powerful decorators for **skipping, parameterizing, marking failures, and defining fixtures**.  
- You can **combine multiple decorators** for complex test scenarios.  

Would you like examples of **custom pytest plugins** or **advanced fixture management**? 🚀