## **1. Interface in Python**  
Python does not have **built-in interfaces** like Java or C#. However, you can achieve interface-like behavior using **abstract base classes (ABCs)** from the `abc` module or **Protocol** from the `typing` module.

---

### **1.1 Using Abstract Base Classes (ABCs)**
Abstract Base Classes **define methods** that must be implemented in derived classes.

```python
from abc import ABC, abstractmethod

class Animal(ABC):  # Abstract class acting as an interface
    @abstractmethod
    def speak(self) -> str:
        pass  # Must be implemented by subclasses

class Dog(Animal):
    def speak(self) -> str:
        return "Woof!"

class Cat(Animal):
    def speak(self) -> str:
        return "Meow!"

dog = Dog()
print(dog.speak())  # Output: Woof!
```

### **Key Points**
- `ABC` (Abstract Base Class) enforces implementation.
- `@abstractmethod` forces subclasses to implement `speak()`.
- You **cannot** instantiate `Animal` directly (`Animal()` would raise an error).

---

### **1.2 Using `Protocol` (Structural Typing)**
Introduced in Python 3.8+, `Protocol` allows defining **interface-like behavior** without requiring explicit inheritance.

```python
from typing import Protocol

class Animal(Protocol):
    def speak(self) -> str:
        ...

class Dog:
    def speak(self) -> str:
        return "Woof!"

class Cat:
    def speak(self) -> str:
        return "Meow!"

def make_noise(animal: Animal) -> None:
    print(animal.speak())

make_noise(Dog())  # Output: Woof!
make_noise(Cat())  # Output: Meow!
```

### **Key Differences: `ABC` vs. `Protocol`**
| Feature | `ABC` (Abstract Class) | `Protocol` |
|---------|----------------|-----------|
| Explicit Inheritance Required | ✅ Yes (`class Dog(Animal)`) | ❌ No (`class Dog:`) |
| Duck Typing | ❌ No | ✅ Yes |
| Runtime Check | ✅ Enforced at runtime | ✅ Enforced at type-checking time (mypy) |

---

## **2. Annotations in Python**
Annotations in Python provide **type hints** but do not enforce types at runtime. They improve **readability, IDE support, and static analysis**.

---

### **2.1 Basic Type Annotations**
```python
def add(x: int, y: int) -> int:
    return x + y
```
- `x: int` → `x` should be an integer.
- `y: int` → `y` should be an integer.
- `-> int` → Function returns an integer.

---

### **2.2 Annotations for Variables**
```python
name: str = "Alice"
age: int = 25
numbers: list[int] = [1, 2, 3]
```

---

### **2.3 Using `Annotated` for Metadata**
Python 3.9+ introduced `Annotated` for **extra metadata**.

```python
from typing import Annotated

def process(value: Annotated[int, "Must be a positive integer"]) -> None:
    print(value)
```

---

### **2.4 Using `get_type_hints()` to Read Annotations**
Annotations are stored in `__annotations__`.

```python
def greet(name: str, age: int) -> str:
    return f"{name} is {age} years old."

print(greet.__annotations__)
# Output: {'name': <class 'str'>, 'age': <class 'int'>, 'return': <class 'str'>}
```

---

## **Summary**
| Concept | Purpose |
|---------|---------|
| **Interface (ABC)** | Enforces method implementation via inheritance |
| **Protocol** | Enforces method implementation using duck typing |
| **Type Annotations** | Provides hints for function parameters and variables |
| **Annotated** | Adds extra metadata to type hints |

Would you like a deep dive into runtime enforcement of annotations?