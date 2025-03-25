In Python, **modules** and **packages** are both ways to organize and reuse code, but they serve different purposes.  

---

## **1. Module**  
A **module** is a single Python file (`.py`) that contains functions, classes, and variables.  

### **Example of a module (`math_utils.py`)**
```python
# math_utils.py
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b
```

### **Importing and using a module**
```python
import math_utils

print(math_utils.add(5, 3))  # Output: 8
```
You can also use:
```python
from math_utils import add
print(add(10, 5))  # Output: 15
```

---

## **2. Package**  
A **package** is a collection of multiple modules inside a directory, and it contains a special file called `__init__.py` (which can be empty or include initialization code). Packages allow for better code organization and modularization.  

### **Example of a package (`mypackage/`)**
```
mypackage/
│── __init__.py
│── math_utils.py
│── string_utils.py
```
- `__init__.py`: Marks the directory as a package.
- `math_utils.py`: Contains mathematical functions.
- `string_utils.py`: Contains string-related functions.

### **Importing from a package**
```python
from mypackage import math_utils
print(math_utils.add(5, 2))  # Output: 7
```
or
```python
from mypackage.math_utils import subtract
print(subtract(9, 4))  # Output: 5
```

---

## **Key Differences**

| Feature       | Module | Package |
|--------------|--------|---------|
| Definition   | A single `.py` file containing functions, classes, or variables. | A directory containing multiple modules and an `__init__.py` file. |
| Organization | Contains related functions or classes. | Groups related modules for better organization. |
| Example      | `math_utils.py` | `mypackage/` (with multiple modules inside) |
| Import       | `import module_name` | `from package_name import module_name` |

---

### **When to Use a Module vs. a Package?**
- Use a **module** when you have a small, self-contained piece of code.
- Use a **package** when you need to organize multiple related modules into a structured directory.

Would you like an example of a real-world package structure?