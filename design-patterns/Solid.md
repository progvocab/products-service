The **SOLID design principles** are a set of guidelines that help developers create software that is maintainable, scalable, and robust. They are often used in the context of object-oriented programming but can also be applied in general software design. Here's an explanation of each principle with examples:

---

### 1. **S - Single Responsibility Principle (SRP)**  
**Definition**: A class should have one, and only one, reason to change.  
- **Goal**: Ensure that a class has a single responsibility, making it easier to maintain and modify.  

**Example**:  
```python
class InvoicePrinter:
    def print_invoice(self, invoice):
        # Code to print the invoice
        pass

class InvoiceCalculator:
    def calculate_total(self, invoice):
        # Code to calculate total
        pass
```
Each class has a single responsibility: `InvoicePrinter` handles printing, and `InvoiceCalculator` handles calculations.

---

### 2. **O - Open/Closed Principle (OCP)**  
**Definition**: A class should be open for extension but closed for modification.  
- **Goal**: You can extend functionality without altering the existing code.  

**Example**:  
```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return 3.14 * self.radius * self.radius
```
Here, you can add new shapes (e.g., `Triangle`) without modifying the existing `Shape` class or its subclasses.

---

### 3. **L - Liskov Substitution Principle (LSP)**  
**Definition**: Objects of a superclass should be replaceable with objects of a subclass without altering the correctness of the program.  
- **Goal**: Ensure that derived classes enhance functionality without breaking the base class contract.  

**Example**:  
```python
class Bird:
    def fly(self):
        return "Flying"

class Sparrow(Bird):
    def fly(self):
        return "Sparrow flying"

class Penguin(Bird):
    def fly(self):
        raise NotImplementedError("Penguins cannot fly")
```
**Fix**: Introduce a new hierarchy to separate flying and non-flying birds:
```python
class Bird:
    pass

class FlyingBird(Bird):
    def fly(self):
        return "Flying"

class Sparrow(FlyingBird):
    pass

class Penguin(Bird):
    pass
```

---

### 4. **I - Interface Segregation Principle (ISP)**  
**Definition**: A class should not be forced to implement interfaces it does not use.  
- **Goal**: Avoid creating large, unwieldy interfaces; instead, create smaller, more specific ones.  

**Example**:  
```python
class Printer:
    def print(self):
        pass

class Scanner:
    def scan(self):
        pass

class MultiFunctionPrinter(Printer, Scanner):
    def print(self):
        # Code for printing
        pass

    def scan(self):
        # Code for scanning
        pass
```
Here, the `Printer` and `Scanner` interfaces are small and specific, so classes implement only what they need.

---

### 5. **D - Dependency Inversion Principle (DIP)**  
**Definition**: High-level modules should not depend on low-level modules. Both should depend on abstractions.  
- **Goal**: Reduce coupling by using abstractions rather than concrete implementations.  

**Example**:  
```python
from abc import ABC, abstractmethod

class NotificationService(ABC):
    @abstractmethod
    def send_notification(self, message):
        pass

class EmailNotification(NotificationService):
    def send_notification(self, message):
        print(f"Sending email: {message}")

class SMSNotification(NotificationService):
    def send_notification(self, message):
        print(f"Sending SMS: {message}")

class NotificationManager:
    def __init__(self, service: NotificationService):
        self.service = service

    def notify(self, message):
        self.service.send_notification(message)

# Usage
email_service = EmailNotification()
manager = NotificationManager(email_service)
manager.notify("Hello, World!")
```
Here, `NotificationManager` depends on the `NotificationService` abstraction, not specific implementations like `EmailNotification` or `SMSNotification`.

---

### Benefits of SOLID Principles
1. **Improved Maintainability**: Each part of the code is easier to understand and modify.
2. **Scalability**: Adding new functionality becomes less risky.
3. **Reduced Coupling**: Code components are loosely coupled, making the system more robust.
4. **Reusability**: Modular and well-defined classes/interfaces can be reused.

By following SOLID principles, you can design flexible, scalable, and maintainable systems.
