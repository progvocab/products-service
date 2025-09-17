### **Writing a Test Class for a Method That Calls an API (Using `requests` and `pytest-mock`)**  

#### **1. API Calling Function**
Let's create a simple function that calls an API using `requests`.  

```python
# api_client.py
import requests

def fetch_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None
```

---

#### **2. Writing a `pytest` Test Class with Mocked API Response**  

We will use `pytest` and `requests-mock` to **mock the API response** instead of making actual network calls.

```python
# test_api_client.py
import pytest
import requests
from unittest.mock import patch
from api_client import fetch_data

class TestApiClient:
    
    @patch("api_client.requests.get")  # Mock the `requests.get` method
    def test_fetch_data_success(self, mock_get):
        mock_response = mock_get.return_value  # Mock response object
        mock_response.status_code = 200
        mock_response.json.return_value = {"message": "success"}
        
        url = "https://api.example.com/data"
        result = fetch_data(url)
        
        assert result == {"message": "success"}
        mock_get.assert_called_once_with(url)  # Ensure API call is made

    @patch("api_client.requests.get")
    def test_fetch_data_failure(self, mock_get):
        mock_response = mock_get.return_value
        mock_response.status_code = 404
        mock_response.json.return_value = {"error": "not found"}
        
        url = "https://api.example.com/data"
        result = fetch_data(url)
        
        assert result is None  # Function should return `None` on failure
        mock_get.assert_called_once_with(url)
```

---

#### **3. Running the Tests**  

Run the test using `pytest`:

```bash
pytest test_api_client.py
```

✅ **Expected Output (If All Tests Pass):**  
```
============================= test session starts =============================
collected 2 items                                                              

test_api_client.py ..                                                     [100%]

============================== 2 passed in 0.01s ==============================
```

---

### **How the Mocking Works**
- `@patch("api_client.requests.get")`: Mocks `requests.get` to prevent actual API calls.
- `mock_get.return_value`: Simulates the API response object.
- `.status_code = 200`: Mocks an HTTP **200 OK** response.
- `.json.return_value = {"message": "success"}`: Mocks the JSON response.
- `mock_get.assert_called_once_with(url)`: Ensures the function called the expected URL.

Would you like to extend this for **handling headers, timeouts, or retries**? 🚀