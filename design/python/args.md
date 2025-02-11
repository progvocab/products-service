### **Positional Arguments (`*args`) and Keyword Arguments (`**kwargs`) in Python**  

Python functions can accept arguments in **different ways**, including **positional arguments**, **keyword arguments**, **default arguments**, and **arbitrary arguments** (`*args`, `**kwargs`).  

---

## **1. Positional Arguments**
Positional arguments are passed in the same **order** as they are defined in the function.

### **1.1 Example of Positional Arguments**
```python
def greet(name, age):
    print(f"Hello, {name}! You are {age} years old.")

greet("Alice", 25)  # ✅ Correct usage
greet(25, "Alice")  # ❌ Incorrect order leads to wrong output
```
- Arguments must be passed in the correct **sequence**.
- `greet(25, "Alice")` will produce incorrect output.

---

## **2. Keyword Arguments**
Keyword arguments allow specifying **parameter names explicitly**, regardless of order.

### **2.1 Example of Keyword Arguments**
```python
def greet(name, age):
    print(f"Hello, {name}! You are {age} years old.")

greet(age=25, name="Alice")  # ✅ Order does not matter
```
- **Key Advantage:** Makes functions more readable and avoids mistakes due to incorrect order.

---

## **3. Default Arguments**
You can set default values for parameters, making them optional.

### **3.1 Example of Default Arguments**
```python
def greet(name, age=30):  # Default age is 30
    print(f"Hello, {name}! You are {age} years old.")

greet("Alice")        # Output: Hello, Alice! You are 30 years old.
greet("Bob", 40)      # Output: Hello, Bob! You are 40 years old.
```
- If no value is provided for `age`, it defaults to `30`.

---

## **4. Arbitrary Positional Arguments (`*args`)**
`*args` allows passing **multiple positional arguments** that are collected as a **tuple**.

### **4.1 Example of `*args`**
```python
def add_numbers(*args):
    return sum(args)

print(add_numbers(1, 2, 3))      # Output: 6
print(add_numbers(10, 20, 30, 40))  # Output: 100
```
- **`args` is a tuple**: `args = (1, 2, 3)`
- Useful when the function needs to handle **variable numbers of arguments**.

---

## **5. Arbitrary Keyword Arguments (`**kwargs`)**
`**kwargs` allows passing multiple **named keyword arguments**, which are collected into a **dictionary**.

### **5.1 Example of `**kwargs`**
```python
def print_info(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print_info(name="Alice", age=25, city="New York")
# Output:
# name: Alice
# age: 25
# city: New York
```
- **`kwargs` is a dictionary**: `kwargs = {"name": "Alice", "age": 25, "city": "New York"}`
- Useful for functions with **flexible named arguments**.

---

## **6. Combining All Argument Types**
Python allows **combining positional, keyword, `*args`, and `**kwargs`**, but there is an **order**:

### **6.1 Correct Order of Parameters**
```python
def function(positional, default=42, *args, keyword, **kwargs):
    print(f"Positional: {positional}")
    print(f"Default: {default}")
    print(f"Args: {args}")
    print(f"Keyword: {keyword}")
    print(f"Kwargs: {kwargs}")

function(1, 99, 2, 3, 4, keyword="hello", extra="yes", more=10)
```
- `1` → `positional`
- `99` → `default`
- `2, 3, 4` → `args`
- `"hello"` → `keyword`
- `{"extra": "yes", "more": 10}` → `kwargs`

**Order Rule:**
1. **Positional arguments** (`arg1, arg2`)
2. **Default arguments** (`arg3=default`)
3. **Arbitrary positional (`*args`)**
4. **Keyword-only arguments** (`keyword`)
5. **Arbitrary keyword arguments (`**kwargs`)**

---

## **7. Keyword-Only and Positional-Only Arguments**
Python allows forcing **some arguments to be positional** or **some to be keyword-only**.

### **7.1 Keyword-Only Arguments (`*` separator)**
Arguments after `*` **must** be passed as keyword arguments.

```python
def greet(name, *, age):
    print(f"Hello, {name}! You are {age} years old.")

greet("Alice", age=30)  # ✅ Correct
greet("Bob", 25)        # ❌ TypeError: age must be a keyword argument
```
- Forces `age` to be passed as a keyword argument (`age=30`).

---

### **7.2 Positional-Only Arguments (`/` separator)**
Introduced in Python 3.8+, arguments before `/` **must** be positional.

```python
def greet(name, age, /):
    print(f"Hello, {name}! You are {age} years old.")

greet("Alice", 30)        # ✅ Correct
greet(name="Bob", age=25) # ❌ TypeError: name must be positional
```
- Forces `name` and `age` to be **positional only**.

---

## **8. Summary Table**
| Type | Example | Description |
|------|---------|-------------|
| **Positional Arguments** | `func(1, 2)` | Order matters |
| **Keyword Arguments** | `func(a=1, b=2)` | Named arguments |
| **Default Arguments** | `func(a=1, b=2, c=10)` | Optional parameters |
| **`*args` (Positional Variable Arguments)** | `func(1, 2, 3, 4)` | Collects multiple arguments as a tuple |
| **`**kwargs` (Keyword Variable Arguments)** | `func(name="Alice", age=25)` | Collects named arguments as a dictionary |
| **Keyword-Only (`*`)** | `func(a, *, b)` | Forces `b` to be keyword-only |
| **Positional-Only (`/`)** | `func(a, b, /)` | Forces `a` and `b` to be positional-only |

Would you like examples of real-world use cases for `*args` and `**kwargs`?