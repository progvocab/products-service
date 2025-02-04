# **Understanding Descriptors in Python**  

## **1. What is a Descriptor?**  
A **descriptor** in Python is an object that customizes the behavior of attribute access (getting, setting, and deleting attributes). Descriptors allow us to define how attributes of a class behave when they are accessed or modified.

### **Use Cases of Descriptors:**
- Implementing **properties** (`property()` is a built-in descriptor).
- Enforcing **validation and type checking**.
- Lazy computation (`@property` with caching).
- **Logging attribute changes**.

---

## **2. Types of Descriptors**
Python descriptors are categorized based on which of the following special methods they implement:

| Descriptor Type | Implements | Description |
|---------------|------------|--------------|
| **Non-Data Descriptor** | `__get__` only | Read-only, cannot prevent attribute assignment. |
| **Data Descriptor** | `__get__`, `__set__`, `__delete__` | Controls both getting and setting attributes. |

---

## **3. Implementing Descriptors with Examples**

### **3.1. Non-Data Descriptor (Read-Only)**
A **non-data descriptor** only defines `__get__()`, meaning it can provide computed values but does not control assignment.

#### **Example: Implementing a Read-Only Descriptor**
```python
class ReadOnlyDescriptor:
    def __get__(self, instance, owner):
        return "This is a read-only value"

class MyClass:
    attribute = ReadOnlyDescriptor()

obj = MyClass()
print(obj.attribute)  # Output: This is a read-only value

obj.attribute = "New Value"  # This will succeed (but doesn't modify descriptor behavior)
print(obj.attribute)  # Output: "New Value" (not controlled by descriptor)
```
✅ **Why?**  
Since `__set__()` is **not defined**, Python allows `attribute` to be overwritten.  

---

### **3.2. Data Descriptor (Enforcing Type Checking)**
A **data descriptor** defines `__get__()` and `__set__()`, meaning it **controls both attribute access and assignment**.

#### **Example: Type-Checked Descriptor**
```python
class Typed:
    def __init__(self, name, expected_type):
        self.name = name
        self.expected_type = expected_type

    def __get__(self, instance, owner):
        return instance.__dict__.get(self.name)

    def __set__(self, instance, value):
        if not isinstance(value, self.expected_type):
            raise TypeError(f"{self.name} must be of type {self.expected_type.__name__}")
        instance.__dict__[self.name] = value

class Person:
    name = Typed("name", str)  # Only strings allowed
    age = Typed("age", int)  # Only integers allowed

p = Person()
p.name = "Alice"  # ✅ Works fine
p.age = 25  # ✅ Works fine
# p.age = "twenty"  # ❌ TypeError: age must be of type int
```
✅ **Why?**  
The `Typed` descriptor **validates types before assignment**.

---

### **3.3. Using `property()` (Built-in Descriptor)**
Python's built-in `property()` function is an example of a descriptor.

#### **Example: Creating a Property**
```python
class Person:
    def __init__(self, name):
        self._name = name

    def get_name(self):
        return self._name

    def set_name(self, value):
        if not isinstance(value, str):
            raise TypeError("Name must be a string")
        self._name = value

    name = property(get_name, set_name)

p = Person("Alice")
print(p.name)  # ✅ Alice
p.name = "Bob"  # ✅ Works fine
# p.name = 123  # ❌ TypeError: Name must be a string
```
✅ **Why?**  
`property()` is a **descriptor that wraps getter and setter methods**.

---

### **3.4. `@property` (Pythonic Way to Create Read-Only Properties)**
Instead of manually using `property()`, we can use `@property`.

#### **Example: Read-Only Property**
```python
class Circle:
    def __init__(self, radius):
        self._radius = radius

    @property
    def area(self):
        return 3.14 * self._radius * self._radius  # Computed on access

c = Circle(5)
print(c.area)  # ✅ 78.5
# c.area = 100  # ❌ AttributeError: can't set attribute
```
✅ **Why?**  
The `@property` decorator makes `area` **computed on the fly** and **immutable**.

---

### **3.5. `@property` with Setter and Deleter**
We can **add setter and deleter methods** using `@<property>.setter` and `@<property>.deleter`.

#### **Example: Read-Write-Delete Property**
```python
class Person:
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        if not isinstance(value, str):
            raise TypeError("Name must be a string")
        self._name = value

    @name.deleter
    def name(self):
        print("Deleting name...")
        del self._name

p = Person("Alice")
print(p.name)  # ✅ Alice
p.name = "Bob"  # ✅ Setter works
del p.name  # ✅ Deleter works (prints: "Deleting name...")
```
✅ **Why?**  
- **Getter:** Fetches `_name`.  
- **Setter:** Validates `name` type.  
- **Deleter:** Removes `_name`.

---

### **3.6. Caching with Descriptors**
If a computed property is expensive, **cache the value** after first computation.

#### **Example: Lazy Loading with Caching**
```python
class CachedProperty:
    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __get__(self, instance, owner):
        if instance not in self.cache:
            self.cache[instance] = self.func(instance)
        return self.cache[instance]

class ExpensiveComputation:
    @CachedProperty
    def compute(self):
        print("Computing result...")
        return 42  # Simulate an expensive operation

obj = ExpensiveComputation()
print(obj.compute)  # ✅ "Computing result..." → 42
print(obj.compute)  # ✅ Cached → 42 (No recomputation)
```
✅ **Why?**  
- First access: **Computes and stores result**.  
- Later access: **Returns cached value**.

---

## **4. Summary of Descriptors**
| Descriptor Type | Methods | Example Use Case |
|---------------|------------|----------------|
| **Non-Data Descriptor** | `__get__` | Read-only attributes (e.g., computed properties) |
| **Data Descriptor** | `__get__`, `__set__`, `__delete__` | Enforcing validation (e.g., type checking, logging) |
| **Built-in Descriptors** | `property()`, `@property` | Built-in property management |
| **Custom Descriptors** | Any of the above | Custom caching, access control |

---

## **Final Thoughts**
- **Use `@property` for simple read-only properties.**
- **Use custom descriptors for complex behavior** (type validation, caching).
- **Descriptors are useful for frameworks like Django, SQLAlchemy, and Airflow**.

Would you like to see **real-world examples from Django or SQLAlchemy**? 🚀