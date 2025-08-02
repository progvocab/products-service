### 🧱 Object-Oriented Programming (OOP) Paradigm

**Object-Oriented Programming (OOP)** is a programming paradigm based on the concept of **"objects"**, which are instances of **classes**. These objects encapsulate **data** (attributes) and **behavior** (methods/functions) and interact with each other to perform tasks.

---

## 🧠 Core Concepts of OOP

| Concept           | Description                                                             |
| ----------------- | ----------------------------------------------------------------------- |
| **Class**         | A blueprint for creating objects (defines structure and behavior)       |
| **Object**        | An instance of a class containing real values                           |
| **Encapsulation** | Bundling data and methods operating on the data within one unit         |
| **Abstraction**   | Hiding implementation details and exposing only relevant features       |
| **Inheritance**   | Creating new classes from existing ones to promote code reuse           |
| **Polymorphism**  | Using a common interface to represent different data types or behaviors |

---

### 🔧 Example in Java

```java
// Class definition
class Animal {
    String name;

    void speak() {
        System.out.println("Animal speaks");
    }
}

// Inheritance and Polymorphism
class Dog extends Animal {
    @Override
    void speak() {
        System.out.println("Dog barks");
    }
}

// Main
public class Main {
    public static void main(String[] args) {
        Animal a = new Dog(); // Polymorphism
        a.speak();            // Output: Dog barks
    }
}
```

---

## ✅ Advantages of OOP

| Advantage           | Description                              |
| ------------------- | ---------------------------------------- |
| **Modularity**      | Code is divided into classes and objects |
| **Reusability**     | Classes can be reused via inheritance    |
| **Scalability**     | Easy to extend the code                  |
| **Maintainability** | Easier to locate and fix issues          |
| **Security**        | Encapsulation hides sensitive data       |

---

## 🧰 Real-World Analogy

Imagine a **Car**:

* **Class**: Blueprint for a car.
* **Object**: Your specific car (`myCar = new Car()`).
* **Encapsulation**: You use the `start()` method, without knowing how the engine starts.
* **Inheritance**: A `SportsCar` class inherits from `Car`.
* **Polymorphism**: A `Car` reference points to a `SportsCar` object and calls the overridden `drive()` method.

---

## ⚙️ Languages That Support OOP

| Language   | OOP Support                                  |
| ---------- | -------------------------------------------- |
| Java       | Full OOP                                     |
| C++        | Full OOP (supports procedural too)           |
| Python     | Multi-paradigm, supports OOP                 |
| C#         | Full OOP                                     |
| Ruby       | Pure OOP                                     |
| JavaScript | Supports OOP with prototype and ES6+ classes |

---

## 📌 When to Use OOP

* When you model real-world entities
* For large codebases needing modularity
* Where reuse, maintainability, and scalability are key
* For GUI apps, game engines, enterprise apps

---

Would you like a comparison between OOP and other paradigms like Functional or Procedural? Or an advanced use case like OOP in Spring Boot with annotations and DI?
