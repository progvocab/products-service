### **Python `typing` Module: A Deep Dive**  
The `typing` module in Python provides **type hints** for static type checking. It helps improve code readability, catches potential errors early, and enhances IDE support.

---

## **1. Basic Type Annotations**
Before diving into advanced types like `Callable` and `Protocol`, here are some **basic type hints**:

```python
from typing import List, Dict, Tuple, Set

def add(x: int, y: int) -> int:
    return x + y

def get_names() -> List[str]:
    return ["Alice", "Bob"]
```

- `int, str, bool, float` → Primitive types.
- `List[str]` → List of strings.
- `Tuple[int, str]` → A tuple with an integer and a string.
- `Dict[str, int]` → Dictionary with string keys and integer values.
- `Set[int]` → A set of integers.

---

## **2. Important Classes in `typing`**

### **2.1 `Callable` (Function Type Hinting)**
**Use Case:** When a function takes another function as an argument.  

```python
from typing import Callable

def apply_function(x: int, func: Callable[[int], int]) -> int:
    return func(x)

def square(n: int) -> int:
    return n * n

print(apply_function(5, square))  # Output: 25
```
- `Callable[[int], int]` means the function takes an `int` and returns an `int`.

---

### **2.2 `Protocol` (Structural Typing)**
**Use Case:** Instead of checking **explicit inheritance**, `Protocol` allows you to define behavior-based typing.

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

def animal_sound(animal: Animal) -> str:
    return animal.speak()

dog = Dog()
cat = Cat()
print(animal_sound(dog))  # Output: Woof!
print(animal_sound(cat))  # Output: Meow!
```
- **Key Benefit:** `Dog` and `Cat` do **not** need to explicitly inherit from `Animal`.  
- **Structural typing** (duck typing) instead of explicit class hierarchy.

---

### **2.3 `TypedDict` (Type-safe Dictionaries)**
**Use Case:** Define dictionaries with **fixed key-value types**.

```python
from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int

def print_person_info(person: Person) -> None:
    print(f"{person['name']} is {person['age']} years old.")

person1 = {"name": "Alice", "age": 30}
print_person_info(person1)
```
- **Key Benefit:** Prevents accidental missing or incorrect keys.

---

### **2.4 `Union` (Multiple Possible Types)**
**Use Case:** When a variable can have **multiple types**.

```python
from typing import Union

def process(value: Union[int, str]) -> str:
    return str(value) + " processed."

print(process(10))    # Output: 10 processed.
print(process("abc")) # Output: abc processed.
```
- **Equivalent to:** `int | str` (Python 3.10+)

---

### **2.5 `Optional` (Nullable Types)**
**Use Case:** When a value **can be `None`**.

```python
from typing import Optional

def greet(name: Optional[str] = None) -> str:
    return f"Hello, {name or 'Guest'}!"

print(greet())        # Output: Hello, Guest!
print(greet("Alice")) # Output: Hello, Alice!
```
- **Equivalent to:** `Union[str, None]`

---

### **2.6 `Any` (Disables Type Checking)**
**Use Case:** When you don’t want type enforcement.

```python
from typing import Any

def process(data: Any) -> None:
    print(data)  # Accepts any type

process(10)
process("text")
process([1, 2, 3])
```
- **Use `Any` cautiously** as it disables type checking.

---

### **2.7 `Literal` (Restrict Values to Specific Constants)**
**Use Case:** When a function should accept **specific values only**.

```python
from typing import Literal

def set_mode(mode: Literal["auto", "manual", "off"]) -> str:
    return f"Mode set to {mode}"

print(set_mode("auto"))  # ✅ Valid
# print(set_mode("random"))  # ❌ Type error
```
- **Key Benefit:** Prevents invalid string inputs.

---

### **2.8 `Final` (Prevents Overriding)**
**Use Case:** Prevent a class or method from being overridden.

```python
from typing import Final

PI: Final = 3.14159  # Cannot be changed
```

```python
from typing import final

class Base:
    @final
    def method(self):
        print("This cannot be overridden")

class Derived(Base):
    pass
    # def method(self):  ❌ Type Error: Cannot override final method
```
- **Key Benefit:** Helps enforce immutability and prevent accidental overrides.

---

### **2.9 `NoReturn` (Indicates a Function Never Returns)**
**Use Case:** When a function **always raises an exception** or never terminates.

```python
from typing import NoReturn

def exit_program() -> NoReturn:
    raise SystemExit("Exiting program...")

# exit_program()  # Raises SystemExit
```

---

## **Summary Table**
| Type Hint | Use Case |
|-----------|---------|
| `List[T]`, `Dict[K, V]`, `Tuple[T, ...]` | Collection types |
| `Callable[[ArgTypes], ReturnType]` | Function arguments as parameters |
| `Protocol` | Structural typing (duck typing) |
| `TypedDict` | Type-safe dictionaries |
| `Union[T1, T2]` | Multiple possible types |
| `Optional[T]` | Value can be `None` |
| `Any` | Disables type checking |
| `Literal["a", "b"]` | Restricts values to specific constants |
| `Final` | Prevents modification or overriding |
| `NoReturn` | Function does not return a value |

Would you like deeper examples on any of these?