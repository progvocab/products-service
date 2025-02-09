### **Scopes in Python: Class, Module, Package, Session, and Others**  

In Python, **scope** defines where a variable or function is accessible. Scopes determine the **visibility and lifetime** of a variable.  

---

## **1️⃣ Built-in Python Scope Types**
Python follows the **LEGB Rule** for resolving variable names:  
1. **Local Scope (L)** – Inside a function.  
2. **Enclosing Scope (E)** – In an enclosing function (nested function).  
3. **Global Scope (G)** – Defined at the top level of a module.  
4. **Built-in Scope (B)** – Includes Python’s built-in functions (e.g., `print()`, `len()`).  

### **Example of LEGB Rule**
```python
x = "global"  # Global scope

def outer():
    x = "enclosing"  # Enclosing scope
    def inner():
        x = "local"  # Local scope
        print(x)  # Local variable is used first
    inner()

outer()  # Output: "local"
print(x)  # Output: "global" (not affected by inner)
```

---

## **2️⃣ Class Scope**
Variables and methods inside a **class** have class-level scope.  
- **Instance variables (`self.variable`)** → Specific to an object instance.  
- **Class variables (`ClassName.variable`)** → Shared by all instances.  

### **Example**
```python
class MyClass:
    class_var = "I am a class variable"

    def __init__(self, value):
        self.instance_var = value  # Instance scope

obj1 = MyClass("Instance 1")
obj2 = MyClass("Instance 2")

print(obj1.instance_var)  # "Instance 1"
print(obj2.instance_var)  # "Instance 2"
print(MyClass.class_var)  # "I am a class variable"
```

---

## **3️⃣ Module Scope**
A **module** is a single Python file (`.py`).  
- **Variables defined at the top level of a module are module-scoped.**  
- **They are accessible within the module but need explicit import in other modules.**  

### **Example**
📌 **File: `module_a.py`**
```python
module_var = "I am a module variable"

def my_function():
    return "Hello from module_a!"
```

📌 **File: `main.py`**
```python
import module_a

print(module_a.module_var)  # "I am a module variable"
print(module_a.my_function())  # "Hello from module_a!"
```

---

## **4️⃣ Package Scope**
A **package** is a directory containing multiple **modules** and an `__init__.py` file.  
- **Package scope is similar to module scope but across multiple files.**  
- **Modules within the same package can access each other using `import package.module`**.  

### **Example**
📂 `my_package/`
```
my_package/
│── __init__.py
│── module1.py
│── module2.py
```

📌 **File: `module1.py`**
```python
def greet():
    return "Hello from module1!"
```

📌 **File: `module2.py`**
```python
from my_package.module1 import greet

print(greet())  # "Hello from module1!"
```

---

## **5️⃣ Session Scope (Used in Testing and Web Apps)**
Session scope refers to **variables that persist across multiple function calls or users during a session**.  
- Used in **pytest**, **Flask/Django sessions**, and **Jupyter Notebook sessions**.  
- Exists until the application restarts or the session expires.  

### **Example in Flask**
```python
from flask import Flask, session

app = Flask(__name__)
app.secret_key = "secret"

@app.route("/set/")
def set_session():
    session["username"] = "Alice"
    return "Session set!"

@app.route("/get/")
def get_session():
    return f"Hello {session.get('username', 'Guest')}!"

# Session persists until browser closes or expires.
```

---

## **6️⃣ Function Scope**
- Variables inside a function are **local** to that function.  
- They **disappear** after the function ends (unless returned or stored globally).  

### **Example**
```python
def my_func():
    local_var = "I exist only inside this function"
    return local_var

print(my_func())  # "I exist only inside this function"
# print(local_var)  # ERROR: local_var is not defined outside the function
```

---

## **7️⃣ Thread Scope**
- Used in **multi-threaded applications**.  
- Each thread has **its own scope** for variables.  

### **Example**
```python
import threading

def print_value():
    thread_local.value = "Thread-specific data"
    print(thread_local.value)

thread_local = threading.local()
thread1 = threading.Thread(target=print_value)
thread2 = threading.Thread(target=print_value)

thread1.start()
thread2.start()
thread1.join()
thread2.join()
```

---

## **8️⃣ Process Scope**
- Variables **persist only within a single process**.  
- Used in **multiprocessing applications**.  

### **Example Using `multiprocessing`**
```python
from multiprocessing import Process, Value

def worker(shared_var):
    shared_var.value += 1
    print(f"Worker process: {shared_var.value}")

shared_var = Value('i', 0)  # Shared integer value
p1 = Process(target=worker, args=(shared_var,))
p2 = Process(target=worker, args=(shared_var,))

p1.start()
p2.start()
p1.join()
p2.join()

print(f"Main process: {shared_var.value}")  # Output may vary due to separate process memory
```

---

## **9️⃣ Global Scope (With `global` Keyword)**
- A variable declared **outside all functions and classes** is **global**.  
- Can be accessed inside a function using `global var_name`.  

### **Example**
```python
x = 10  # Global variable

def modify_global():
    global x
    x += 5  # Modify the global variable
    print(x)  # 15

modify_global()
print(x)  # 15
```

---

## **🔟 Nonlocal Scope (For Nested Functions)**
- Used in **nested functions** to modify an **enclosing (non-global) variable**.  

### **Example**
```python
def outer():
    x = "outer variable"

    def inner():
        nonlocal x
        x = "modified inside inner"
        print(x)

    inner()
    print(x)  # "modified inside inner"

outer()
```

---

## **📝 Summary of Scopes**
| Scope | Where It Exists | Lifetime |
|-------|---------------|---------|
| **Local** | Inside functions | Function execution |
| **Enclosing** | Nested functions | Outer function execution |
| **Global** | Module level | Until program ends |
| **Built-in** | Python built-ins (`print()`) | Always available |
| **Class** | Inside a class | Object lifetime |
| **Module** | Inside a module (`.py`) | Until module is unloaded |
| **Package** | Across multiple modules | Until package is removed |
| **Session** | Web session (Flask/Django) | Until session expires |
| **Thread** | Inside a thread | Until thread ends |
| **Process** | Inside a process | Until process ends |
| **Nonlocal** | Enclosing function variables | Until outer function ends |

Would you like examples for a **specific use case** like **pytest fixtures, Django, or multiprocessing**? 🚀
