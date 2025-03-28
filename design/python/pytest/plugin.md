### **Creating a Custom Pytest Plugin with Hooks**
Let's create a **custom Pytest plugin** that:
1. **Adds a CLI option (`--slow`)** to selectively run slow tests.
2. **Logs test durations** after each test runs.
3. **Generates a custom test report file**.

---

## **Step 1: Create a Plugin File (`pytest_custom_plugin.py`)**
This file defines our **plugin hooks**.

```python
import pytest
import time

# 1️⃣ Add a CLI option to enable slow tests
def pytest_addoption(parser):
    parser.addoption("--slow", action="store_true", help="Run slow tests")

# 2️⃣ Skip slow tests if --slow is not provided
@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    if not config.getoption("--slow"):
        skip_slow = pytest.mark.skip(reason="Skipped because --slow is not set")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

# 3️⃣ Measure test execution time
@pytest.hookimpl
def pytest_runtest_call(item):
    start_time = time.time()
    yield  # Run the actual test
    duration = time.time() - start_time
    item.user_properties.append(("duration", duration))

# 4️⃣ Write test results to a custom report file
@pytest.hookimpl
def pytest_sessionfinish(session, exitstatus):
    with open("test_report.txt", "w") as f:
        f.write(f"Pytest finished with exit code {exitstatus}\n")
        for item in session.items:
            duration = dict(item.user_properties).get("duration", None)
            if duration:
                f.write(f"Test {item.nodeid} ran in {duration:.2f} seconds\n")
```

---

## **Step 2: Create a Sample Test File (`test_example.py`)**
This file contains tests, including one marked as `@pytest.mark.slow`.

```python
import pytest
import time

def test_fast():
    assert 1 + 1 == 2

@pytest.mark.slow
def test_slow():
    time.sleep(2)  # Simulating a slow test
    assert 2 * 2 == 4
```

---

## **Step 3: Run Tests with and Without `--slow`**
1. **Run all tests (skips slow tests by default):**
   ```bash
   pytest --tb=short
   ```
   Output:
   ```
   SKIPPED [1] test_example.py: Skipped because --slow is not set
   ```

2. **Run all tests including slow tests:**
   ```bash
   pytest --slow --tb=short
   ```

3. **Check the generated test report (`test_report.txt`)**
   ```bash
   cat test_report.txt
   ```
   Example content:
   ```
   Pytest finished with exit code 0
   Test test_example.py::test_fast ran in 0.00 seconds
   Test test_example.py::test_slow ran in 2.00 seconds
   ```

---

## **Summary**
✅ **Adds a CLI option (`--slow`)** to control slow test execution.  
✅ **Tracks test execution times** using `pytest_runtest_call()`.  
✅ **Generates a custom report** in `test_report.txt`.  

Would you like help packaging this as a **reusable Pytest plugin**?

### **Packaging a Custom Pytest Plugin as a Reusable Package**  
Now, let's package our **Pytest plugin** so it can be **installed and reused** in multiple projects.

---

## **Step 1: Create a Directory Structure**
```bash
my_pytest_plugin/
│── pytest_custom_plugin/
│   │── __init__.py
│   │── hooks.py
│── tests/
│   │── test_example.py
│── setup.py
│── pyproject.toml
│── README.md
```

---

## **Step 2: Move the Plugin Code into `hooks.py`**  
Create the file: **`pytest_custom_plugin/hooks.py`**  

```python
import pytest
import time

# 1️⃣ Add a CLI option --slow
def pytest_addoption(parser):
    parser.addoption("--slow", action="store_true", help="Run slow tests")

# 2️⃣ Modify test collection to skip slow tests unless --slow is used
@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    if not config.getoption("--slow"):
        skip_slow = pytest.mark.skip(reason="Skipped because --slow is not set")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

# 3️⃣ Measure test execution time
@pytest.hookimpl
def pytest_runtest_call(item):
    start_time = time.time()
    yield  # Run the test
    duration = time.time() - start_time
    item.user_properties.append(("duration", duration))

# 4️⃣ Write test results to a report file
@pytest.hookimpl
def pytest_sessionfinish(session, exitstatus):
    with open("test_report.txt", "w") as f:
        f.write(f"Pytest finished with exit code {exitstatus}\n")
        for item in session.items:
            duration = dict(item.user_properties).get("duration", None)
            if duration:
                f.write(f"Test {item.nodeid} ran in {duration:.2f} seconds\n")
```

---

## **Step 3: Register the Plugin in `__init__.py`**  
Create the file: **`pytest_custom_plugin/__init__.py`**  

```python
# Register the plugin with Pytest
pytest_plugins = ["pytest_custom_plugin.hooks"]
```

---

## **Step 4: Create `setup.py` to Package the Plugin**
Create the file: **`setup.py`**  

```python
from setuptools import setup, find_packages

setup(
    name="pytest-custom-plugin",
    version="0.1",
    packages=find_packages(),
    entry_points={"pytest11": ["custom_plugin = pytest_custom_plugin.hooks"]},
    install_requires=["pytest"],
    author="Your Name",
    description="A Pytest plugin to handle slow tests and custom reports",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Framework :: Pytest",
        "Programming Language :: Python :: 3",
    ],
)
```

---

## **Step 5: Create `pyproject.toml` (Optional for Modern Packaging)**
Create the file: **`pyproject.toml`**  

```toml
[build-system]
requires = ["setuptools", "wheel"]
build-backend = "setuptools.build_meta"
```

---

## **Step 6: Install the Plugin Locally**
Navigate to the **`my_pytest_plugin/`** directory and run:
```bash
pip install .
```
This installs the **pytest-custom-plugin** as a local package.

---

## **Step 7: Use the Plugin in a Test Project**
Now, in **any test suite**, you can use this plugin.

### **Example Test File (`tests/test_example.py`)**
```python
import pytest
import time

def test_fast():
    assert 1 + 1 == 2

@pytest.mark.slow
def test_slow():
    time.sleep(2)  # Simulating a slow test
    assert 2 * 2 == 4
```

---

## **Step 8: Run Tests Using the Plugin**
1. **Run tests normally (skips slow tests by default)**:
   ```bash
   pytest
   ```
   Output:
   ```
   SKIPPED [1] test_example.py: Skipped because --slow is not set
   ```

2. **Run all tests including slow tests**:
   ```bash
   pytest --slow
   ```

3. **Check the test report (`test_report.txt`)**:
   ```bash
   cat test_report.txt
   ```
   Example Output:
   ```
   Pytest finished with exit code 0
   Test tests/test_example.py::test_fast ran in 0.00 seconds
   Test tests/test_example.py::test_slow ran in 2.00 seconds
   ```

---

## **Step 9: Publish the Plugin to PyPI (Optional)**
1. Install the **build tools**:
   ```bash
   pip install build twine
   ```
2. Build the package:
   ```bash
   python -m build
   ```
3. Upload to **PyPI**:
   ```bash
   twine upload dist/*
   ```

---

## **Summary**
✅ **Creates a reusable Pytest plugin**  
✅ **Skips slow tests unless `--slow` is used**  
✅ **Logs test execution time**  
✅ **Generates a custom report file**  
✅ **Packaged and installable as a Python package**  

Would you like help with **adding more features** to the plugin (e.g., test retries, logging enhancements)?