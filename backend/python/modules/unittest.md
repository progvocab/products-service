## **`unittest` Package in Python**

The `unittest` package in Python is a built-in **unit testing framework** used to write and execute test cases for code validation. It is inspired by JUnit (used in Java) and follows the **object-oriented approach** to testing.

---

## **1. Key Features of `unittest`**
- **Test Case Creation** using `unittest.TestCase`
- **Assertions** to verify expected outputs
- **Setup & Teardown** methods (`setUp()` and `tearDown()`)
- **Test Discovery** for running multiple tests
- **Mocking** support (`unittest.mock`)
- **Test Execution and Reporting** using `unittest.main()`

---

## **2. Writing a Basic Test Case**
Let's test a simple function that adds two numbers.

### **Function to Test**
```python
# calculator.py
def add(a, b):
    return a + b
```

### **Test Case Using `unittest`**
```python
# test_calculator.py
import unittest
from calculator import add

class TestCalculator(unittest.TestCase):

    def test_add(self):
        self.assertEqual(add(2, 3), 5)
        self.assertEqual(add(-1, 1), 0)
        self.assertEqual(add(0, 0), 0)
        self.assertEqual(add(10, 20), 30)

if __name__ == '__main__':
    unittest.main()
```

### **Run the Test**
```bash
python test_calculator.py
```

✅ **Expected Output (if all tests pass):**
```
.
----------------------------------------------------------------------
Ran 1 test in 0.001s

OK
```

---

## **3. Important Assertions in `unittest`**
| Assertion | Description |
|-----------|-------------|
| `assertEqual(a, b)` | Check if `a == b` |
| `assertNotEqual(a, b)` | Check if `a != b` |
| `assertTrue(x)` | Check if `x` is `True` |
| `assertFalse(x)` | Check if `x` is `False` |
| `assertIs(a, b)` | Check if `a is b` |
| `assertIsNot(a, b)` | Check if `a is not b` |
| `assertIsNone(x)` | Check if `x is None` |
| `assertIsNotNone(x)` | Check if `x is not None` |
| `assertRaises(Exception, func, *args)` | Check if `func(*args)` raises an exception |

### **Example: Testing Exceptions**
```python
def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

class TestCalculator(unittest.TestCase):
    def test_divide_by_zero(self):
        with self.assertRaises(ValueError):
            divide(10, 0)
```

---

## **4. Using `setUp()` and `tearDown()` for Test Setup**
- `setUp()`: Runs **before** every test case.
- `tearDown()`: Runs **after** every test case.

### **Example**
```python
class TestExample(unittest.TestCase):
    def setUp(self):
        self.data = [1, 2, 3]  # Setup common test data

    def tearDown(self):
        self.data = None  # Cleanup after test

    def test_length(self):
        self.assertEqual(len(self.data), 3)
```

---

## **5. Running Multiple Tests Automatically**
You can run all test files automatically using:

```bash
python -m unittest discover
```

Or run a **specific test file**:
```bash
python -m unittest test_calculator.py
```

---

## **6. Mocking API Calls Using `unittest.mock`**
Use `unittest.mock` to **mock external API calls** and prevent real HTTP requests.

```python
from unittest.mock import patch
import requests

def fetch_data(url):
    response = requests.get(url)
    return response.json()

class TestAPI(unittest.TestCase):
    @patch("requests.get")
    def test_fetch_data(self, mock_get):
        mock_get.return_value.json.return_value = {"message": "success"}
        self.assertEqual(fetch_data("https://api.example.com"), {"message": "success"})
```

---

## **7. Conclusion**
- `unittest` is a powerful **built-in** testing framework in Python.
- Supports **test discovery, assertions, setup/teardown, and mocking**.
- Easily integrated into **CI/CD pipelines** for automation.

Would you like to see how `unittest` integrates with **pytest or CI/CD tools**? 🚀