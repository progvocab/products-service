### **Using Fixtures in Pytest**  

**Pytest fixtures** are used to set up and tear down test dependencies efficiently. They help create reusable test setups without duplication.

---

## **1. What is a Fixture?**
A **fixture** in `pytest`:
- Provides test setup (e.g., database connections, test data).
- Can be **scoped** (`function`, `class`, `module`, `session`).
- Can be **parameterized** to run tests with multiple configurations.

---

## **2. Basic Example of a Pytest Fixture**
A fixture returning **test data**:

```python
import pytest

# Define a fixture
@pytest.fixture
def sample_data():
    return {"name": "Alice", "age": 30}

# Use the fixture in a test
def test_sample_data(sample_data):
    assert sample_data["name"] == "Alice"
    assert sample_data["age"] == 30
```
### **How It Works?**
- The `@pytest.fixture` decorator creates a reusable **fixture**.
- The test function **requests** the fixture (`sample_data`), and pytest **injects** the return value.

Run the test:
```bash
pytest test_sample.py -v
```
✅ **Expected Output**:
```
test_sample.py::test_sample_data PASSED
```

---

## **3. Using Fixtures for Test Setup & Teardown**
Use `yield` for **setup and cleanup**:

```python
import pytest

@pytest.fixture
def setup_and_teardown():
    print("\nSetup: Creating test environment")
    yield {"status": "ready"}  # This value is returned to the test function
    print("\nTeardown: Cleaning up after test")

def test_example(setup_and_teardown):
    assert setup_and_teardown["status"] == "ready"
```
### **Execution Flow:**
1. **Before the test** → "Setup" runs.
2. **Test runs** → Uses fixture data.
3. **After the test** → "Teardown" runs.

---

## **4. Fixture Scope (`function`, `class`, `module`, `session`)**
- `function` (default) → Runs **for each test**.
- `class` → Runs **once per class**.
- `module` → Runs **once per module**.
- `session` → Runs **once per test session**.

### **Example: Module-Level Fixture**
```python
@pytest.fixture(scope="module")
def db_connection():
    print("\nOpening Database Connection")
    yield "DB Connection"
    print("\nClosing Database Connection")

def test_db_query1(db_connection):
    assert db_connection == "DB Connection"

def test_db_query2(db_connection):
    assert db_connection == "DB Connection"
```
- **Setup runs once for all tests in the module**.
- **Teardown runs after all tests complete**.

---

## **5. Parameterized Fixtures**
Run a test with multiple data sets.

```python
@pytest.fixture(params=[("Alice", 30), ("Bob", 25)])
def user_data(request):
    return request.param

def test_user_age(user_data):
    name, age = user_data
    assert isinstance(name, str)
    assert isinstance(age, int)
```
✅ **Runs twice**: once for **Alice (30)**, once for **Bob (25)**.

---

## **6. Using Fixtures in a Test Class**
```python
class TestUser:
    @pytest.fixture(scope="class")
    def user(self):
        return {"name": "Charlie", "age": 40}

    def test_name(self, user):
        assert user["name"] == "Charlie"

    def test_age(self, user):
        assert user["age"] == 40
```
- The **fixture is shared across all test methods** in the class.

---

## **7. Mocking API Calls with Fixtures**
Fixtures can be used with `pytest-mock` to **mock external services**.

```python
import pytest
from unittest.mock import MagicMock
import requests

@pytest.fixture
def mock_requests_get(mocker):
    mock = mocker.patch("requests.get")
    mock.return_value.json.return_value = {"message": "success"}
    return mock

def test_fetch_data(mock_requests_get):
    response = requests.get("https://api.example.com").json()
    assert response == {"message": "success"}
```
- `mocker.patch("requests.get")` replaces `requests.get` with a **mock**.
- Prevents real API calls and ensures predictable responses.

---

## **8. Running Pytest with Fixtures**
Run all tests:
```bash
pytest -v
```
Run tests matching a specific fixture:
```bash
pytest -v -k "test_sample_data"
```

---

## **Conclusion**
- ✅ **Fixtures remove redundant test setup code.**
- ✅ **They support setup & teardown automatically.**
- ✅ **They can be scoped, parameterized, and used for mocking.**
- ✅ **Enhance test maintainability and efficiency.**

Would you like an example integrating **database connections** with fixtures? 🚀