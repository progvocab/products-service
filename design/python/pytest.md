# **Pytest Decorators in Python – A Detailed Guide with Examples**  

**Pytest** is a powerful testing framework in Python, and **decorators** in pytest allow us to modify test behavior dynamically.  

---

## **1. What are Pytest Decorators?**  
Pytest **decorators** are special functions prefixed with `@pytest.` that allow us to **mark, skip, parameterize, or modify** test execution.  

Common pytest decorators include:
1. `@pytest.mark.skip` → Skip a test.  
2. `@pytest.mark.skipif(condition, reason="...")` → Skip a test conditionally.  
3. `@pytest.mark.xfail` → Mark a test as expected to fail.  
4. `@pytest.mark.parametrize` → Run a test multiple times with different inputs.  
5. `@pytest.fixture` → Define reusable test setup functions.  

---

## **2. Pytest Decorator Examples**  

### **2.1. Skipping Tests (`@pytest.mark.skip`)**  
Use this when you want to **temporarily skip a test**.  

```python
import pytest

@pytest.mark.skip(reason="This test is not ready yet")
def test_example():
    assert 2 + 2 == 4
```
- **Output:** Test is skipped with a message: `"This test is not ready yet"`.

---

### **2.2. Conditional Skipping (`@pytest.mark.skipif`)**  
Skip a test based on a **specific condition** (e.g., OS, Python version).  

```python
import sys
import pytest

@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires Python 3.8+")
def test_python_version():
    assert sys.version_info >= (3, 8)
```
- This test **runs only on Python 3.8 or higher**.

---

### **2.3. Marking Tests as Expected to Fail (`@pytest.mark.xfail`)**  
If a test is **expected to fail**, use `@pytest.mark.xfail` instead of removing it.  

```python
import pytest

@pytest.mark.xfail(reason="Known issue with division by zero")
def test_divide_by_zero():
    assert 1 / 0 == 0
```
- **Output:** Pytest marks this test as **expected to fail** instead of treating it as an error.

---

### **2.4. Parameterized Tests (`@pytest.mark.parametrize`)**  
Run the **same test multiple times** with different inputs.  

```python
import pytest

@pytest.mark.parametrize("x, y, expected", [(2, 3, 5), (1, 5, 6), (0, 0, 0)])
def test_addition(x, y, expected):
    assert x + y == expected
```
- This test runs **three times**, once for each set of parameters.

---

### **2.5. Using Fixtures (`@pytest.fixture`)**  
Fixtures **setup and teardown** reusable test data.  

```python
import pytest

@pytest.fixture
def sample_data():
    return {"name": "Alice", "age": 30}

def test_data(sample_data):
    assert sample_data["name"] == "Alice"
    assert sample_data["age"] == 30
```
- The `sample_data` fixture is **automatically passed** to the test function.

---

### **2.6. Custom Markers (`@pytest.mark.<custom>`)**  
Define **custom test categories**.  

```python
import pytest

@pytest.mark.slow
def test_large_computation():
    assert sum(range(100000)) > 0
```
Run only slow tests using:  
```bash
pytest -m slow
```

---

## **3. Combining Multiple Pytest Decorators**  
Decorators can be **stacked** for more complex scenarios.  

```python
import pytest
import sys

@pytest.mark.parametrize("x, y, expected", [(2, 3, 5), (1, 1, 2)])
@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires Python 3.8+")
def test_addition(x, y, expected):
    assert x + y == expected
```
- **Runs multiple cases** but **skips if Python < 3.8**.

---

## **4. Running Pytest with Decorators**  
Run tests with:
```bash
pytest -v
```
Run **only marked tests**:
```bash
pytest -m slow
```
Run **skipped tests**:
```bash
pytest --runxfail
```

---

## **Conclusion**  
Pytest decorators provide **powerful control** over test execution. They allow you to **skip, parameterize, mark expected failures, and define reusable test setups** using fixtures.  

Would you like to see advanced pytest usage, such as **custom plugins or reporting**? 🚀