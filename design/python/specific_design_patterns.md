### **Python-Specific Design Patterns with Code Examples**
Python has **unique design patterns** that leverage its **dynamic nature, metaprogramming, and built-in features**. Below are some important **Python-specific patterns** along with **detailed explanations and code examples**.

---

## **1. Duck Typing Pattern**
### **Concept**:
- Python follows **"duck typing"**, meaning **"if it looks like a duck and quacks like a duck, it must be a duck."**
- Instead of checking types explicitly, Python relies on **object behavior**.
- This allows flexibility in function arguments without enforcing inheritance.

### **Example: Duck Typing in Action**
```python
class Dog:
    def speak(self):
        return "Woof!"

class Cat:
    def speak(self):
        return "Meow!"

class Robot:
    def speak(self):
        return "Beep Boop!"

def make_speak(animal):
    print(animal.speak())  # No type check, just calling the method

# Different objects, same interface
make_speak(Dog())     # Output: Woof!
make_speak(Cat())     # Output: Meow!
make_speak(Robot())   # Output: Beep Boop!
```
✅ **Use When**: You want **flexibility without enforcing inheritance**.

---

## **2. Context Manager Pattern**
### **Concept**:
- Used to **manage resources efficiently** (e.g., files, database connections).
- Implements `__enter__` and `__exit__` methods.
- Automatically **closes resources** using the `with` statement.

### **Example: File Handling with Context Manager**
```python
class FileOpener:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode

    def __enter__(self):
        self.file = open(self.filename, self.mode)
        return self.file  # Returns file object

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()  # Closes file automatically

# Using 'with' statement
with FileOpener("test.txt", "w") as f:
    f.write("Hello, world!")

# No need to explicitly close the file
```
✅ **Use When**: You need **automatic cleanup of resources**.

### **Built-in Alternative**
```python
with open("test.txt", "w") as f:
    f.write("Hello, world!")
```
Python **already provides built-in context managers** for files, sockets, and database connections.

---

## **3. Metaclass Pattern**
### **Concept**:
- **Metaclasses control how classes are created**.
- Used for **enforcing rules, modifying class attributes**, or **tracking classes dynamically**.
- The `type` function is used to define metaclasses.

### **Example: Enforcing Class Naming Convention**
```python
class NameEnforcerMeta(type):
    def __new__(cls, name, bases, class_dict):
        if not name[0].isupper():
            raise TypeError(f"Class name '{name}' must start with an uppercase letter.")
        return super().__new__(cls, name, bases, class_dict)

class Person(metaclass=NameEnforcerMeta):
    pass

# class person(metaclass=NameEnforcerMeta):  # ❌ TypeError: Class name 'person' must start with an uppercase letter
#     pass
```
✅ **Use When**: You need to **enforce rules on class creation**.

---

## **4. Monkey Patching Pattern**
### **Concept**:
- **Modifies or extends existing classes dynamically at runtime**.
- Often used in **testing, debugging, or adding new features to libraries**.
- Be **careful** as it can make debugging harder.

### **Example: Patching a Method in an Existing Class**
```python
class Vehicle:
    def move(self):
        return "Moving..."

# Patch the method at runtime
def fly(self):
    return "Flying instead!"

Vehicle.move = fly  # Monkey patching

v = Vehicle()
print(v.move())  # Output: Flying instead!
```
✅ **Use When**: You need to **modify behavior at runtime** (e.g., during testing).

### **Example: Monkey Patching in Unit Tests**
```python
import time

def slow_function():
    time.sleep(5)  # Simulates a long process
    return "Done"

# Monkey patch for testing
def fast_function():
    return "Instant result"

slow_function = fast_function  # Patching

print(slow_function())  # Output: Instant result
```
✅ **Use When**: You need to **replace slow functions during testing**.

---

## **5. Lazy Initialization Pattern**
### **Concept**:
- **Delays object creation until it's actually needed**.
- Saves memory and speeds up application startup.

### **Example: Lazy Loading in a Database Connection**
```python
class Database:
    _connection = None  # Initially, no connection

    def connect(self):
        if self._connection is None:  # Create only if needed
            print("Initializing database connection...")
            self._connection = "Connected"
        return self._connection

db = Database()
print(db.connect())  # Output: Initializing database connection...
print(db.connect())  # Output: Connected
```
✅ **Use When**: You want to **optimize resource usage**.

---

## **6. Registry Pattern**
### **Concept**:
- **Keeps track of all available classes/functions dynamically**.
- Useful for **plugin systems, event handling, and dependency injection**.

### **Example: Plugin System Using Registry**
```python
class Registry:
    _registry = {}

    @classmethod
    def register(cls, name, func):
        cls._registry[name] = func

    @classmethod
    def execute(cls, name, *args):
        if name in cls._registry:
            return cls._registry[name](*args)
        else:
            raise ValueError(f"No function registered under '{name}'")

# Register functions dynamically
Registry.register("greet", lambda name: f"Hello, {name}!")
Registry.register("add", lambda x, y: x + y)

# Execute registered functions
print(Registry.execute("greet", "Alice"))  # Output: Hello, Alice!
print(Registry.execute("add", 5, 3))       # Output: 8
```
✅ **Use When**: You need **dynamic function/class registration**.

---

## **7. Memoization Pattern**
### **Concept**:
- **Caches results of expensive function calls** to **avoid redundant calculations**.
- Python’s `functools.lru_cache` provides built-in support.

### **Example: Fibonacci with Memoization**
```python
from functools import lru_cache

@lru_cache(maxsize=None)  # Infinite cache size
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

print(fibonacci(10))  # Output: 55 (cached results for faster execution)
```
✅ **Use When**: You need to **optimize recursive functions**.

---

## **Summary Table**
| **Pattern** | **Purpose** | **Usage Example** |
|------------|------------|--------------------|
| **Duck Typing** | Uses object behavior instead of types. | `make_speak(animal)` |
| **Context Manager** | Manages resource cleanup. | `with open()` |
| **Metaclass** | Controls class creation. | `metaclass=NameEnforcerMeta` |
| **Monkey Patching** | Modifies behavior at runtime. | `Vehicle.move = fly` |
| **Lazy Initialization** | Delays object creation. | `db.connect()` |
| **Registry** | Tracks registered classes/functions. | `Registry.register("greet", lambda x: x)` |
| **Memoization** | Caches expensive function calls. | `@lru_cache` |

Would you like more **real-world use cases** for any pattern?