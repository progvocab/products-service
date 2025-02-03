# **Common Design Patterns in Python (with Examples)**  

**Design patterns** are reusable solutions to common software design problems. They help **organize code, improve maintainability, and follow best practices**. Python supports several **design patterns**, classified into **three main types**:

1. **Creational Patterns** (For object creation)
2. **Structural Patterns** (For organizing code)
3. **Behavioral Patterns** (For managing object interactions)

---

## **1. Creational Design Patterns**  
These patterns deal with object creation **while keeping code flexible and reusable**.

### **1.1. Singleton Pattern**  
✅ **Ensures only one instance of a class exists** throughout the program.  

```python
class Singleton:
    _instance = None  # Class-level variable to store the instance

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

# Test Singleton
obj1 = Singleton()
obj2 = Singleton()
print(obj1 is obj2)  # Output: True (both refer to the same instance)
```

**Use Case:** Logging, Configuration Management, Database Connection Pool.

---

### **1.2. Factory Pattern**  
✅ **Creates objects without specifying the exact class**.  

```python
class Dog:
    def speak(self):
        return "Woof!"

class Cat:
    def speak(self):
        return "Meow!"

def animal_factory(animal_type):
    animals = {"dog": Dog, "cat": Cat}
    return animals.get(animal_type, Dog)()  # Default to Dog if not found

# Test Factory
pet = animal_factory("cat")
print(pet.speak())  # Output: Meow!
```

**Use Case:** When the object type is determined dynamically at runtime.

---

## **2. Structural Design Patterns**  
These patterns help **structure classes and objects efficiently**.

### **2.1. Adapter Pattern**  
✅ **Allows incompatible interfaces to work together** by acting as a bridge.

```python
class EuropeanPlug:
    def power_220v(self):
        return "220V Power"

class USPlugAdapter:
    def __init__(self, european_plug):
        self.european_plug = european_plug

    def power_110v(self):
        return f"Converted to 110V -> {self.european_plug.power_220v()}"

# Test Adapter
plug = EuropeanPlug()
adapter = USPlugAdapter(plug)
print(adapter.power_110v())  # Output: Converted to 110V -> 220V Power
```

**Use Case:** Converting APIs, handling legacy code.

---

### **2.2. Decorator Pattern**  
✅ **Dynamically adds behavior to an object** without modifying its structure.

```python
def uppercase_decorator(func):
    def wrapper():
        result = func()
        return result.upper()
    return wrapper

@uppercase_decorator
def greet():
    return "hello, world"

print(greet())  # Output: HELLO, WORLD
```

**Use Case:** Logging, Authentication, API Rate-Limiting.

---

### **2.3. Proxy Pattern**  
✅ **Controls access to an object** (like security or lazy initialization).

```python
class RealDatabase:
    def query(self):
        return "Fetching Data"

class DatabaseProxy:
    def __init__(self):
        self._real_db = None

    def query(self):
        if self._real_db is None:
            self._real_db = RealDatabase()  # Lazy Initialization
        return self._real_db.query()

# Test Proxy
proxy = DatabaseProxy()
print(proxy.query())  # Output: Fetching Data
```

**Use Case:** Lazy loading, security access control.

---

## **3. Behavioral Design Patterns**  
These patterns **manage object interactions** and behaviors.

### **3.1. Observer Pattern**  
✅ **Notifies multiple objects when a state changes** (Publisher-Subscriber).

```python
class Subject:
    def __init__(self):
        self._observers = []

    def attach(self, observer):
        self._observers.append(observer)

    def notify(self, message):
        for observer in self._observers:
            observer.update(message)

class Observer:
    def update(self, message):
        print(f"Received update: {message}")

# Test Observer
subject = Subject()
obs1 = Observer()
obs2 = Observer()

subject.attach(obs1)
subject.attach(obs2)
subject.notify("New Event!")  # Both observers receive the message
```

**Use Case:** Event-driven systems, GUI applications.

---

### **3.2. Strategy Pattern**  
✅ **Allows selecting an algorithm at runtime**.

```python
class Strategy:
    def execute(self, a, b):
        pass

class Add(Strategy):
    def execute(self, a, b):
        return a + b

class Multiply(Strategy):
    def execute(self, a, b):
        return a * b

class Context:
    def __init__(self, strategy):
        self.strategy = strategy

    def execute_strategy(self, a, b):
        return self.strategy.execute(a, b)

# Test Strategy
context = Context(Add())
print(context.execute_strategy(5, 3))  # Output: 8

context.strategy = Multiply()
print(context.execute_strategy(5, 3))  # Output: 15
```

**Use Case:** Payment Methods, Sorting Algorithms.

---

### **3.3. Command Pattern**  
✅ **Encapsulates a request as an object**, allowing undo/redo operations.

```python
class Command:
    def execute(self):
        pass

class LightOn(Command):
    def execute(self):
        print("Light turned ON")

class RemoteControl:
    def __init__(self, command):
        self.command = command

    def press_button(self):
        self.command.execute()

# Test Command Pattern
light_on = LightOn()
remote = RemoteControl(light_on)
remote.press_button()  # Output: Light turned ON
```

**Use Case:** GUI Buttons, Undo-Redo functionality.

---

### **3.4. Chain of Responsibility Pattern**  
✅ **Passes requests along a chain of handlers**.

```python
class Handler:
    def __init__(self, next_handler=None):
        self.next_handler = next_handler

    def handle(self, request):
        if self.next_handler:
            return self.next_handler.handle(request)
        return "Request Not Handled"

class AuthHandler(Handler):
    def handle(self, request):
        if request == "auth":
            return "Authenticated!"
        return super().handle(request)

class LoggingHandler(Handler):
    def handle(self, request):
        if request == "log":
            return "Logging Event!"
        return super().handle(request)

# Test Chain of Responsibility
chain = AuthHandler(LoggingHandler())
print(chain.handle("log"))  # Output: Logging Event!
```

**Use Case:** Logging, Authentication Pipelines.

---

## **Summary Table**
| **Pattern** | **Type** | **Use Case** |
|------------|---------|-------------|
| **Singleton** | Creational | Database connection, Configuration |
| **Factory** | Creational | Object creation based on input |
| **Adapter** | Structural | API integration, Legacy code |
| **Decorator** | Structural | Logging, Function enhancements |
| **Proxy** | Structural | Lazy loading, Security access control |
| **Observer** | Behavioral | Event-driven programming, Notifications |
| **Strategy** | Behavioral | Selecting algorithms dynamically |
| **Command** | Behavioral | GUI actions, Undo/Redo |
| **Chain of Responsibility** | Behavioral | Request handling in a pipeline |

---

## **Conclusion**
Design patterns **simplify complex systems** by making them more structured and reusable. Python’s flexibility makes it **easy to implement these patterns efficiently**.

Would you like me to focus on **a specific pattern in more detail**?