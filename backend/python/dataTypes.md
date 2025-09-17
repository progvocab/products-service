### **Data Types in Python**  
Python has various built-in data types categorized into numbers, sequences, sets, mappings, and more.

---

## **1. Numeric Types**
| Data Type | Description | Example |
|-----------|------------|---------|
| **`int`** | Integer numbers (positive, negative, zero) | `x = 10` |
| **`float`** | Decimal (floating-point) numbers | `y = 3.14` |
| **`complex`** | Complex numbers (real + imaginary) | `z = 2 + 3j` |

### **Example:**
```python
a = 5      # int
b = 2.5    # float
c = 1 + 2j # complex
print(type(a), type(b), type(c))
```

---

## **2. Sequence Types (Ordered Collections)**
| Data Type | Description | Example |
|-----------|------------|---------|
| **`str`** | Text (string) | `"Hello"` |
| **`list`** | Ordered, mutable collection | `[1, 2, 3]` |
| **`tuple`** | Ordered, immutable collection | `(1, 2, 3)` |

### **Example:**
```python
s = "Python"         # str
lst = [1, 2, 3, 4]   # list
tup = (10, 20, 30)   # tuple
print(type(s), type(lst), type(tup))
```

---

## **3. Set Types (Unordered Collections)**
| Data Type | Description | Example |
|-----------|------------|---------|
| **`set`** | Unordered, unique values | `{1, 2, 3}` |
| **`frozenset`** | Immutable set | `frozenset({1, 2, 3})` |

### **Example:**
```python
s = {1, 2, 3, 4}  # set
fs = frozenset(s) # frozenset
print(type(s), type(fs))
```

---

## **4. Mapping Type (Key-Value Pairs)**
| Data Type | Description | Example |
|-----------|------------|---------|
| **`dict`** | Key-value pairs (mutable) | `{"name": "Alice", "age": 25}` |

### **Example:**
```python
d = {"name": "Alice", "age": 25}
print(type(d))
```

---

## **5. Boolean Type**
| Data Type | Description | Example |
|-----------|------------|---------|
| **`bool`** | `True` or `False` | `x = True` |

### **Example:**
```python
x = True
y = False
print(type(x), type(y))
```

---

## **6. Binary Types**
| Data Type | Description | Example |
|-----------|------------|---------|
| **`bytes`** | Immutable byte sequence | `b"hello"` |
| **`bytearray`** | Mutable byte sequence | `bytearray([65, 66, 67])` |
| **`memoryview`** | Memory-efficient view of bytes | `memoryview(b"hello")` |

### **Example:**
```python
b = bytes("hello", "utf-8")
ba = bytearray(b)
mv = memoryview(b)
print(type(b), type(ba), type(mv))
```

---

## **7. None Type**
| Data Type | Description | Example |
|-----------|------------|---------|
| **`NoneType`** | Represents "nothing" or "no value" | `x = None` |

### **Example:**
```python
x = None
print(type(x))
```

---

## **8. User-Defined Data Types**
Python allows defining custom data types using **classes**.
```python
class Person:
    def __init__(self, name):
        self.name = name

p = Person("John")
print(type(p))
```

---

## **9. Type Checking and Conversion**
### **Checking Data Types**
```python
print(type(42))        # <class 'int'>
print(isinstance(3.14, float))  # True
```

### **Converting Between Types**
```python
print(int(3.14))      # 3
print(float(10))      # 10.0
print(str(100))       # "100"
print(list("abc"))    # ['a', 'b', 'c']
print(set([1, 2, 2, 3]))  # {1, 2, 3}
```

---

## **Summary**
| Category | Data Types |
|----------|-----------|
| **Numbers** | `int`, `float`, `complex` |
| **Sequences** | `str`, `list`, `tuple` |
| **Sets** | `set`, `frozenset` |
| **Mapping** | `dict` |
| **Boolean** | `bool` |
| **Binary** | `bytes`, `bytearray`, `memoryview` |
| **None** | `NoneType` |
| **User-Defined** | `class` |

Would you like deeper explanations of any specific type?