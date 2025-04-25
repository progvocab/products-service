# **Python Decorators Explained with Examples**

## **1. What is a Decorator in Python?**
A **decorator** in Python is a function that **modifies the behavior of another function or class without changing its code**. Decorators use the `@decorator_name` syntax and are often used for logging, authentication, timing, and more.

✅ **Key Features of Decorators:**
- Functions in Python are **first-class objects** (can be passed as arguments).
- Decorators allow us to **extend functionality** dynamically.
- They use **closures** to wrap another function.

---

## **2. Basic Example of a Decorator**
Let's start with a simple decorator that prints a message **before and after** calling a function.

```python
def my_decorator(func):
    def wrapper():
        print("Before the function call")
        func()
        print("After the function call")
    return wrapper

@my_decorator  # Applying the decorator
def say_hello():
    print("Hello, World!")

say_hello()
```

### **Output:**
```
Before the function call
Hello, World!
After the function call
```
✅ The `@my_decorator` modifies `say_hello()` **without changing its code**.

---

## **3. Decorators with Arguments**
If the decorated function takes arguments, the wrapper function must handle them.

```python
def repeat(func):
    def wrapper(*args, **kwargs):
        print("Function will run twice:")
        func(*args, **kwargs)
        func(*args, **kwargs)
    return wrapper

@repeat
def greet(name):
    print(f"Hello, {name}!")

greet("Alice")
```

### **Output:**
```
Function will run twice:
Hello, Alice!
Hello, Alice!
```
✅ `*args, **kwargs` allow handling any number of arguments dynamically.

---

## **4. Real-World Examples of Decorators**
### **4.1. Logging Decorator**
A decorator to log function calls.

```python
import time

def log_function(func):
    def wrapper(*args, **kwargs):
        print(f"Calling function: {func.__name__} with {args}, {kwargs}")
        result = func(*args, **kwargs)
        print(f"{func.__name__} returned {result}")
        return result
    return wrapper

@log_function
def add(a, b):
    return a + b

add(3, 5)
```

### **Output:**
```
Calling function: add with (3, 5), {}
add returned 8
```
✅ Useful for debugging.

---

### **4.2. Timing Decorator**
Measure execution time of a function.

```python
def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper

@timer
def slow_function():
    time.sleep(2)
    print("Function finished")

slow_function()
```

### **Output:**
```
Function finished
slow_function executed in 2.0001 seconds
```
✅ Helps optimize performance.

---

### **4.3. Authentication Decorator**
Restricts function access.

```python
def requires_auth(func):
    def wrapper(user):
        if user != "admin":
            print("Access Denied!")
            return
        return func(user)
    return wrapper

@requires_auth
def admin_dashboard(user):
    print(f"Welcome, {user}! Access granted.")

admin_dashboard("guest")  # Access Denied
admin_dashboard("admin")  # Access Granted
```

### **Output:**
```
Access Denied!
Welcome, admin! Access granted.
```
✅ Used in web apps to protect routes.

---

## **5. Class-Based Decorators**
Decorators can also be implemented using **classes** instead of functions.

```python
class MyDecorator:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        print("Class-based Decorator: Before function call")
        result = self.func(*args, **kwargs)
        print("Class-based Decorator: After function call")
        return result

@MyDecorator
def say_hello():
    print("Hello, World!")

say_hello()
```

### **Output:**
```
Class-based Decorator: Before function call
Hello, World!
Class-based Decorator: After function call
```
✅ Useful when maintaining state inside the decorator.

---

## **6. Nested Decorators (Multiple Decorators)**
You can apply multiple decorators to a function.

```python
def decorator1(func):
    def wrapper():
        print("Decorator 1")
        func()
    return wrapper

def decorator2(func):
    def wrapper():
        print("Decorator 2")
        func()
    return wrapper

@decorator1
@decorator2
def my_function():
    print("Hello from function")

my_function()
```

### **Output:**
```
Decorator 1
Decorator 2
Hello from function
```
✅ Order matters: `@decorator1` wraps `@decorator2`, which wraps `my_function()`.

---

## **7. Built-in Python Decorators**
Python provides some built-in decorators:

| Decorator | Description |
|-----------|-------------|
| `@staticmethod` | Defines a static method inside a class. |
| `@classmethod` | Defines a method that takes `cls` as the first argument. |
| `@property` | Defines a method as a property (getter). |

### **Example: Using `@property`**
```python
class Car:
    def __init__(self, model, price):
        self.model = model
        self._price = price  # Private attribute

    @property
    def price(self):
        return self._price

car = Car("Tesla", 50000)
print(car.price)  # Calls the getter method
```

✅ `@property` allows **accessing a method like an attribute**.

---

## **8. When to Use Decorators?**
✔ **Logging function calls**  
✔ **Measuring execution time**  
✔ **Enforcing authentication**  
✔ **Modifying behavior dynamically**  
✔ **Code reusability & readability**

---

## **Conclusion**
- **Decorators enhance functions without modifying their core logic**.
- They are useful in real-world applications like **logging, security, caching, and profiling**.
- Both **function-based and class-based decorators** are widely used.

Would you like me to explain decorators in **Flask, Django, or FastAPI** as well?