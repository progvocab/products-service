Here’s a complete explanation of the most widely used and recognized **design principles** in software development, especially in **object-oriented programming (OOP)**, along with concise **Java code examples** for each.

---

## ✅ 1. **SOLID Principles**

The SOLID principles are five foundational principles for writing clean, maintainable OOP code.

---

### 🔹 S — Single Responsibility Principle (SRP)

> A class should have only one reason to change.

```java
class Invoice {
    public void calculateTotal() { /* logic */ }
}

class InvoicePrinter {
    public void printInvoice(Invoice invoice) { /* logic */ }
}
```

✅ `Invoice` handles calculations
✅ `InvoicePrinter` handles printing

---

### 🔹 O — Open/Closed Principle (OCP)

> Software entities should be open for extension but closed for modification.

```java
interface Shape {
    double area();
}

class Circle implements Shape {
    public double area() { return Math.PI * 5 * 5; }
}

class Square implements Shape {
    public double area() { return 4 * 4; }
}
```

✅ New shapes can be added without changing existing code

---

### 🔹 L — Liskov Substitution Principle (LSP)

> Subtypes must be substitutable for their base types.

```java
class Bird {
    public void fly() { System.out.println("Flying"); }
}

class Sparrow extends Bird {}
// A Penguin class that can't fly would violate LSP if used here
```

✅ Subclasses should behave consistently with the parent class's expectations

---

### 🔹 I — Interface Segregation Principle (ISP)

> No client should be forced to depend on methods it does not use.

```java
interface Printer {
    void print();
}

interface Scanner {
    void scan();
}

class MultiFunctionPrinter implements Printer, Scanner {
    public void print() {}
    public void scan() {}
}
```

✅ Break interfaces into smaller, role-specific contracts

---

### 🔹 D — Dependency Inversion Principle (DIP)

> High-level modules should not depend on low-level modules; both should depend on abstractions.

```java
interface Keyboard {
    void input();
}

class WiredKeyboard implements Keyboard {
    public void input() {}
}

class Computer {
    private Keyboard keyboard;
    public Computer(Keyboard keyboard) {
        this.keyboard = keyboard;
    }
}
```

✅ `Computer` depends on `Keyboard` interface, not a specific implementation

---

## 🧰 2. DRY – Don’t Repeat Yourself

> Avoid duplication in logic, code, or data.

❌ Bad:

```java
System.out.println("User name: " + user.getName());
System.out.println("User email: " + user.getEmail());
```

✅ Better:

```java
void printUser(User user) {
    System.out.println("User name: " + user.getName());
    System.out.println("User email: " + user.getEmail());
}
```

---

## 🧼 3. KISS – Keep It Simple, Stupid

> Prefer simplicity over cleverness.

```java
// ❌ Overcomplicated
if ((x & 1) == 1) {}

// ✅ Simpler
if (x % 2 != 0) {}
```

---

## 🔮 4. YAGNI – You Aren’t Gonna Need It

> Don’t implement something until it’s necessary.

```java
// ❌ Don’t add future-proof features you may not need
class Report {
    public void exportToPDF() {}
    // public void exportToXML() {} ← not needed now
}
```

---

## 🔁 5. Separation of Concerns

> Divide responsibilities into distinct sections.

```java
class AuthService {
    public boolean login(String user, String pass) { return true; }
}

class UserController {
    AuthService auth = new AuthService();
    public void handleLogin() {
        auth.login("u", "p");
    }
}
```

✅ Business logic is separate from controller/UI logic

---

## 📏 6. Law of Demeter (Principle of Least Knowledge)

> A method should only interact with its immediate dependencies.

❌ Bad:

```java
order.getCustomer().getAddress().getCity();
```

✅ Better:

```java
order.getCustomerCity();
```

---

## 📚 Summary Table

| Principle      | Description                                 | Java Code Example                   |
| -------------- | ------------------------------------------- | ----------------------------------- |
| SRP            | One class = One reason to change            | `Invoice` + `InvoicePrinter`        |
| OCP            | Open for extension, closed for modification | `Shape` interface                   |
| LSP            | Subtypes should behave like parent types    | `Bird` / `Sparrow`                  |
| ISP            | Prefer smaller interfaces                   | `Printer` / `Scanner`               |
| DIP            | Depend on abstractions, not implementations | `Computer(Keyboard)`                |
| DRY            | Avoid duplicated logic                      | `printUser()`                       |
| KISS           | Keep logic simple                           | `% 2 != 0` instead of bit ops       |
| YAGNI          | Don’t add features preemptively             | Avoid `exportToXML()` unless needed |
| SoC            | Keep layers independent                     | `AuthService` vs `Controller`       |
| Law of Demeter | Talk to friends, not strangers              | `order.getCustomerCity()`           |

---

Would you like all these principles as a GitHub-ready sample project with one file per principle?
