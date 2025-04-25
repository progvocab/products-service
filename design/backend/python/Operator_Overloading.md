### **Operator Overloading in Python**
Operator overloading allows you to redefine how operators (`+`, `-`, `*`, etc.) behave for user-defined classes by implementing special methods (also called **dunder methods**, like `__add__`, `__sub__`, etc.).

---

## **Example 1: Overloading `+` (Addition Operator)**
By default, `+` works on numbers and strings. We can overload it for a custom class.

### **Without Overloading**
```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

p1 = Point(2, 3)
p2 = Point(4, 5)

# p1 + p2 would cause an error
```
⛔ **Error:** `"unsupported operand type(s) for +"`

### **With Overloading**
```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __str__(self):
        return f"({self.x}, {self.y})"

p1 = Point(2, 3)
p2 = Point(4, 5)

p3 = p1 + p2  # Calls __add__()
print(p3)  # Output: (6, 8)
```
✅ Now `+` works for `Point` objects.

---

## **Example 2: Overloading `*` (Multiplication Operator)**
Let's define multiplication for a vector class.

```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)

    def __str__(self):
        return f"({self.x}, {self.y})"

v = Vector(2, 3)
result = v * 3  # Calls __mul__()
print(result)  # Output: (6, 9)
```
✅ Now `*` works for scalar multiplication.

---

## **Example 3: Overloading `==` (Equality Operator)**
By default, `==` compares object memory locations. We can define custom behavior.

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __eq__(self, other):
        return self.name == other.name and self.age == other.age

p1 = Person("Alice", 25)
p2 = Person("Alice", 25)
p3 = Person("Bob", 30)

print(p1 == p2)  # True (same name & age)
print(p1 == p3)  # False (different attributes)
```
✅ Now `==` checks for logical equality instead of memory location.

---

## **Example 4: Overloading `-` (Subtraction Operator)**
```python
class Number:
    def __init__(self, value):
        self.value = value

    def __sub__(self, other):
        return Number(self.value - other.value)

    def __str__(self):
        return str(self.value)

n1 = Number(10)
n2 = Number(4)

print(n1 - n2)  # Output: 6
```
✅ Now `-` subtracts values inside our class.

---

## **Example 5: Overloading `>` (Greater Than Operator)**
```python
class Student:
    def __init__(self, name, marks):
        self.name = name
        self.marks = marks

    def __gt__(self, other):
        return self.marks > other.marks

s1 = Student("Alice", 85)
s2 = Student("Bob", 90)

print(s1 > s2)  # False
print(s2 > s1)  # True
```
✅ Now `>` compares student marks.

---

## **Example 6: Overloading `[]` (Indexing Operator)**
```python
class ShoppingCart:
    def __init__(self, items):
        self.items = items

    def __getitem__(self, index):
        return self.items[index]

cart = ShoppingCart(["Apple", "Banana", "Cherry"])
print(cart[1])  # Output: Banana
```
✅ Now we can use `[]` to access elements.

---

## **Example 7: Overloading `()` (Callable Objects)**
```python
class Counter:
    def __init__(self, count=0):
        self.count = count

    def __call__(self):
        self.count += 1
        return self.count

c = Counter()
print(c())  # Output: 1
print(c())  # Output: 2
```
✅ Now instances act like functions!

---

### **Conclusion**
- **Operator overloading** makes custom objects behave more like built-in types.
- Use **dunder methods** (`__add__`, `__mul__`, `__getitem__`, etc.).
- Improves **code readability** and **natural usage** of objects.

Would you like an example for a specific operator? 🚀
