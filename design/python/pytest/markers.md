### **Pytest Markers: A Detailed Guide with Examples**  

Pytest **markers** are used to **categorize, organize, and control test execution**. They help in **running specific test groups**, **skipping tests**, **parameterizing tests**, and more.  

---

## **1. Types of Pytest Markers**
| Marker | Description | Example |
|--------|------------|---------|
| `@pytest.mark.skip` | Skips a test unconditionally | `@pytest.mark.skip(reason="Not implemented yet")` |
| `@pytest.mark.skipif(condition, reason)` | Skips test if a condition is met | `@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires Python 3.8")` |
| `@pytest.mark.xfail` | Marks a test as expected to fail | `@pytest.mark.xfail(reason="Known bug")` |
| `@pytest.mark.slow` | Custom marker for slow tests | `@pytest.mark.slow` |
| `@pytest.mark.parametrize` | Runs the test multiple times with different inputs | `@pytest.mark.parametrize("input, expected", [(2, 4), (3, 9)])` |
| `@pytest.mark.usefixtures` | Applies a fixture to a test function/class | `@pytest.mark.usefixtures("setup_data")` |

---

## **2. Common Pytest Markers with Examples**

### **(a) `@pytest.mark.skip` - Skipping a Test**  
Use when a test is not relevant or needs to be skipped for now.

```python
import pytest

@pytest.mark.skip(reason="Skipping due to incomplete implementation")
def test_example():
    assert 1 + 1 == 2
```
**Run:**
```bash
pytest -v
```
**Output:**
```
SKIPPED [1] test_file.py:2: Skipping due to incomplete implementation
```

---

### **(b) `@pytest.mark.skipif(condition, reason)` - Conditional Skipping**  
Skip a test **only if a certain condition is met**.

```python
import pytest
import sys

@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires Python 3.8 or higher")
def test_python_version():
    assert sys.version_info >= (3, 8)
```
**Run on Python 3.7:**
```
SKIPPED [1] test_file.py: Skipping due to Python version
```

---

### **(c) `@pytest.mark.xfail` - Expected Failure**  
Mark a test as **expected to fail**, useful for **known bugs**.

```python
import pytest

@pytest.mark.xfail(reason="Bug in progress")
def test_bug():
    assert 1 + 1 == 3  # This will fail
```
**Output:**
```
XFAIL test_file.py::test_bug
```

---

### **(d) `@pytest.mark.parametrize` - Running Tests with Multiple Inputs**  
Run the same test **multiple times** with different parameters.

```python
import pytest

@pytest.mark.parametrize("num, square", [(2, 4), (3, 9), (4, 16)])
def test_square(num, square):
    assert num ** 2 == square
```
**Output:**
```
test_file.py::test_square[2-4] PASSED
test_file.py::test_square[3-9] PASSED
test_file.py::test_square[4-16] PASSED
```

---

### **(e) Custom Markers (e.g., `@pytest.mark.slow`)**  
You can define **custom markers** to group tests.

#### **Step 1: Define a Custom Marker (`slow`)**
```python
import pytest

@pytest.mark.slow
def test_large_dataset():
    assert sum(range(1000000)) == 499999500000
```
#### **Step 2: Register the Custom Marker in `pytest.ini`**
Create a `pytest.ini` file in the root of your project:
```ini
[pytest]
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
```
#### **Step 3: Run Only `slow` Tests**
```bash
pytest -m slow
```

---

### **(f) `@pytest.mark.usefixtures` - Apply Fixtures to Tests**  
Use fixtures without explicitly passing them.

```python
import pytest

@pytest.fixture
def setup_data():
    print("\nSetting up test data...")

@pytest.mark.usefixtures("setup_data")
def test_sample():
    assert 1 + 1 == 2
```
**Output:**
```
Setting up test data...
test_sample PASSED
```

---

## **3. Running Tests with Markers**
| Command | Description |
|---------|-------------|
| `pytest -m slow` | Run only tests marked with `@pytest.mark.slow` |
| `pytest -m "not slow"` | Run all tests **except** `slow` tests |
| `pytest -m "slow or integration"` | Run tests marked as `slow` or `integration` |

---

## **Conclusion**
Pytest markers allow you to:
- **Skip tests** (`skip`, `skipif`)
- **Mark failing tests** (`xfail`)
- **Run tests with multiple inputs** (`parametrize`)
- **Group and filter tests** (custom markers like `slow`)

Would you like a **hands-on example** using markers in a real project?