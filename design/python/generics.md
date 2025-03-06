### **Generics in Python**
Generics in Python allow you to write **type-safe, reusable, and flexible code** by defining **generic types** that work with multiple data types. Python introduced **generics** with **type hints** using the `typing` module.

---

## **1. Why Use Generics?**
- Avoids **code duplication** by allowing a function/class to work with **multiple types**.
- Improves **type safety** by ensuring correct type usage.
- Enhances **code readability** and helps with **static type checking**.

---

## **2. Generics with Functions**
You can use **generics** with functions using the `TypeVar` class.

### **Example: A Generic Function**
```python
from typing import TypeVar

T = TypeVar("T")  # Declare a generic type variable

def repeat(value: T, times: int) -> list[T]:
    return [value] * times  # Returns a list of repeated values

print(repeat(5, 3))       # Output: [5, 5, 5] (integers)
print(repeat("Hi", 2))    # Output: ['Hi', 'Hi'] (strings)
```
✅ **How It Works?**  
- `TypeVar("T")` allows us to define a **generic type T**.
- `repeat(value: T, times: int) -> list[T]` ensures that `value` is of **any type** and returns a `list[T]` (a list of the same type).
- The function works with **both integers and strings** without changing the code.

---

## **3. Generics with Classes**
### **Example: A Generic Stack Class**
```python
from typing import Generic, TypeVar

T = TypeVar("T")  # Declare a generic type variable

class Stack(Generic[T]):  # Define a generic class
    def __init__(self):
        self._items: list[T] = []  # List of type T

    def push(self, item: T) -> None:
        self._items.append(item)

    def pop(self) -> T:
        return self._items.pop()

    def peek(self) -> T:
        return self._items[-1]

    def is_empty(self) -> bool:
        return len(self._items) == 0

# Integer stack
int_stack = Stack[int]()
int_stack.push(10)
int_stack.push(20)
print(int_stack.pop())  # Output: 20

# String stack
str_stack = Stack[str]()
str_stack.push("Hello")
str_stack.push("World")
print(str_stack.pop())  # Output: World
```
✅ **How It Works?**  
- `Stack(Generic[T])` makes `Stack` a **generic class** that works with any type.
- `Stack[int]()` creates an **integer stack**, while `Stack[str]()` creates a **string stack**.

---

## **4. Generics with Multiple Type Variables**
You can define **multiple type variables** when dealing with **mixed data types**.

### **Example: Generic Key-Value Store**
```python
from typing import TypeVar, Generic

K = TypeVar("K")  # Key type
V = TypeVar("V")  # Value type

class KeyValueStore(Generic[K, V]):
    def __init__(self):
        self.store: dict[K, V] = {}  # Dictionary of key-value pairs

    def add(self, key: K, value: V) -> None:
        self.store[key] = value

    def get(self, key: K) -> V:
        return self.store[key]

kv = KeyValueStore[int, str]()
kv.add(1, "One")
kv.add(2, "Two")
print(kv.get(1))  # Output: One
```
✅ **How It Works?**  
- `KeyValueStore(Generic[K, V])` makes it a **generic dictionary-like class**.
- Works with **any key-value type**, e.g., `KeyValueStore[int, str]()`.

---

## **5. Generics with Inheritance**
Generics can also be used in **inheritance** to create flexible hierarchies.

### **Example: Generic Animal Class**
```python
from typing import TypeVar, Generic

T = TypeVar("T")

class Animal(Generic[T]):
    def __init__(self, sound: T):
        self.sound = sound

    def make_sound(self) -> T:
        return self.sound

class Dog(Animal[str]):  # Dog has sound of type str
    pass

dog = Dog("Woof")
print(dog.make_sound())  # Output: Woof
```
✅ **How It Works?**  
- `Animal(Generic[T])` makes `Animal` a **generic base class**.
- `Dog(Animal[str])` specifies that **Dog uses `str` for sound**.

---

## **6. Generics with Type Boundaries**
Sometimes, you want to **restrict** a generic type to **specific types**.

### **Example: Restricting Type to Numbers**
```python
from typing import TypeVar
from numbers import Number

T = TypeVar("T", bound=Number)  # Restrict T to Number types

def square(value: T) -> T:
    return value * value  # Only works for numbers

print(square(4))      # Output: 16
print(square(3.5))    # Output: 12.25
# print(square("Hello"))  # ❌ Type Error: str is not a Number
```
✅ **How It Works?**  
- `TypeVar("T", bound=Number)` ensures `T` **must be a number**.
- This prevents **accidental passing of invalid types**.

---

## **7. Generics with Protocols (Structural Typing)**
Python **does not require explicit interfaces**, but we can use **Protocols** to define **expected behaviors**.

### **Example: Generic Protocol for Objects with a Length**
```python
from typing import Protocol, TypeVar

class Sized(Protocol):
    def __len__(self) -> int: ...

T = TypeVar("T", bound=Sized)  # Restrict T to objects with __len__()

def print_length(obj: T) -> None:
    print(f"Length: {len(obj)}")

print_length([1, 2, 3])  # ✅ Works (list has __len__())
print_length("Hello")    # ✅ Works (str has __len__())
# print_length(42)       # ❌ TypeError: int has no __len__()
```
✅ **How It Works?**  
- `Sized(Protocol)` defines a **behavior (having `__len__`)**.
- `T = TypeVar("T", bound=Sized)` ensures `T` **must have `__len__()`**.

---

## **Summary Table**
| **Feature** | **Description** | **Example** |
|------------|----------------|------------|
| **Generic Function** | Works with multiple data types. | `repeat(value: T, times: int)` |
| **Generic Class** | A class that works with different types. | `Stack[T]` |
| **Multiple TypeVars** | Supports multiple generic types. | `KeyValueStore[K, V]` |
| **Generics in Inheritance** | Allows base classes to be generic. | `Animal[T]` |
| **Type Boundaries** | Restricts generics to a specific type. | `TypeVar("T", bound=Number)` |
| **Structural Typing (Protocol)** | Ensures objects implement required behavior. | `Sized(Protocol)` |

Would you like **more examples** on a specific use case?