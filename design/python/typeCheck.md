You can implement **type safety** and **type checking** in Python function arguments using:  

1. **Type hints (PEP 484)**
2. **Runtime type checking with `mypy`**
3. **Using `pydantic` for stricter validation**
4. **Using `functools.singledispatch` for function overloading**

---

### **1. Type Hints (Static Type Checking)**
Python allows **type hints**, but they are not enforced at runtime. However, tools like `mypy` can check for type correctness.  
```python
def add_numbers(a: int, b: int) -> int:
    return a + b

# Valid call
print(add_numbers(5, 3))

# Invalid call (detected by mypy, but Python will still run it)
print(add_numbers("5", 3))  # TypeError if mypy is used
```
Run `mypy` to check for type errors:  
```sh
mypy script.py
```

---

### **2. Enforcing Type Checking at Runtime**
You can use **`isinstance`** inside functions to manually enforce type safety.  
```python
def add_numbers(a: int, b: int) -> int:
    if not isinstance(a, int) or not isinstance(b, int):
        raise TypeError("Both arguments must be integers")
    return a + b

# add_numbers("5", 3)  # Raises TypeError
```

---

### **3. Using `pydantic` for Strict Type Validation**
`pydantic` is a powerful library for runtime validation.  
```python
from pydantic import BaseModel

class Numbers(BaseModel):
    a: int
    b: int

def add_numbers(data: Numbers) -> int:
    return data.a + data.b

# Valid input
print(add_numbers(Numbers(a=5, b=3)))

# add_numbers(Numbers(a="5", b=3))  # Raises ValidationError
```
Install `pydantic`:  
```sh
pip install pydantic
```

---

### **4. Function Overloading with `functools.singledispatch`**
For handling multiple types dynamically:  
```python
from functools import singledispatch

@singledispatch
def process_data(value):
    raise TypeError("Unsupported type")

@process_data.register
def _(value: int):
    return f"Processing int: {value}"

@process_data.register
def _(value: str):
    return f"Processing string: {value}"

print(process_data(10))      # Processing int: 10
print(process_data("hello")) # Processing string: hello
# print(process_data(3.5))    # Raises TypeError
```

---

### **Which Method to Use?**
- **Static checking (`mypy`)** → Best for large codebases  
- **Runtime checking (`isinstance`)** → Useful for critical validation  
- **`pydantic`** → Best for APIs & data validation  
- **`functools.singledispatch`** → Best for function overloading  

Would you like type safety enforced in your **Jenkins pipeline scripts** or Python-based **data processing**?