### **`pytest.hookimpl` in Pytest**
The `pytest.hookimpl` decorator is used to **override or modify** Pytest's built-in hooks. Pytest follows a **hook-based** architecture, where plugins and test suites can register implementations of certain predefined hooks.

#### **Why Use `pytest.hookimpl`?**
- Modify Pytest behavior (e.g., logging, custom test filtering, reporting).
- Extend Pytest functionality through plugins.
- Hook into the test lifecycle to execute custom code before/after tests.

---

## **Basic Example: Custom Logging Before Tests Run**
```python
import pytest

@pytest.hookimpl(tryfirst=True)  # Runs before any other hook
def pytest_runtest_setup(item):
    print(f"\nSetting up test: {item.name}")
```
🔹 **How It Works:**
- The `pytest_runtest_setup` hook runs before each test.
- `item.name` gives the test function name.
- The `tryfirst=True` ensures it runs **before other implementations**.

---

## **Example: Skipping Tests Based on a Condition**
Let's skip tests if an environment variable is missing.
```python
import pytest
import os

@pytest.hookimpl(trylast=True)  # Runs after other setup hooks
def pytest_runtest_setup(item):
    if "MY_ENV_VAR" not in os.environ:
        pytest.skip(f"Skipping {item.name} as MY_ENV_VAR is not set")
```
🔹 **How It Works:**
- Checks if `MY_ENV_VAR` is set.
- If not, **skips the test dynamically**.

---

## **Example: Customizing Test Reports**
Modify the **test result output** by implementing `pytest_runtest_logreport`.
```python
@pytest.hookimpl(tryfirst=True)
def pytest_runtest_logreport(report):
    if report.when == "call" and report.failed:
        print(f"❌ Test Failed: {report.nodeid}")
    elif report.when == "call" and report.passed:
        print(f"✅ Test Passed: {report.nodeid}")
```
🔹 **How It Works:**
- Runs after a test completes.
- Checks if the test **failed or passed**.
- Customizes output based on result.

---

## **Example: Registering a Custom CLI Option**
Add a `--custom-flag` option to Pytest.
```python
def pytest_addoption(parser):
    parser.addoption("--custom-flag", action="store_true", help="Enable custom behavior")

@pytest.hookimpl
def pytest_configure(config):
    if config.getoption("--custom-flag"):
        print("Custom Flag Enabled!")
```
🔹 **How It Works:**
- `pytest_addoption`: Adds a `--custom-flag` CLI option.
- `pytest_configure`: Runs at startup and checks if the flag is enabled.

---

## **Summary**
| **Hook**                     | **Purpose** |
|------------------------------|------------|
| `pytest_runtest_setup`       | Run setup code before each test. |
| `pytest_runtest_logreport`   | Customize test result logs. |
| `pytest_addoption`           | Add custom CLI options. |
| `pytest_configure`           | Configure behavior at startup. |

### **Would you like an example of a complete Pytest plugin using hooks?**