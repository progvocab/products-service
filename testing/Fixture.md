A **fixture** in software testing is any **fixed set of data, state, or environment setup** that tests rely on.

In simple terms:
👉 A fixture is **the “setup” you need before running a test** (and sometimes the “cleanup” after).

---

### Why do we need fixtures?

* To ensure **tests start from a known state**.
* To **reuse setup code** across multiple tests (avoid duplication).
* To make tests **deterministic and reliable**.

---

### Examples of Fixtures

#### 1. **Unit Testing Fixture**

In JUnit (Java):

```java
public class CalculatorTest {
    private Calculator calculator;

    @BeforeEach
    void setUp() { // Fixture setup
        calculator = new Calculator();
    }

    @Test
    void testAdd() {
        assertEquals(5, calculator.add(2, 3));
    }
}
```

Here:

* `@BeforeEach setUp()` = **fixture** that ensures every test starts with a new `Calculator`.

---

#### 2. **Integration Test Fixture**

* Spinning up an **in-memory database** before tests.
* Preloading tables with known test data.
* Shutting it down after tests.

---

#### 3. **Pytest Fixture** (Python)

```python
import pytest

@pytest.fixture
def sample_data():
    return {"user": "Alice", "age": 30}

def test_user(sample_data):
    assert sample_data["user"] == "Alice"
```

Here `sample_data` is a **fixture** that provides consistent test input.

---

### Kinds of Fixtures

1. **Data Fixtures** → test data (mock records, JSON files).
2. **Environment Fixtures** → database connections, servers, files.
3. **State Fixtures** → initializing objects, clearing caches, resetting configs.
4. **Cleanup Fixtures** → tearDown/reset after tests.

---

✅ **In short:**
A fixture is **the preparation and cleanup code** that gives your tests a stable, repeatable starting point. Without fixtures, tests might be flaky or dependent on each other.

---

Do you want me to show you **how fixtures are used in OpenSearch tests** specifically?
