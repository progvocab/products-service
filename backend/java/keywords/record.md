### **`record` Keyword in Java (Introduced in Java 14 - Preview, Stable in Java 16)**  

In Java, the `record` keyword is used to **create immutable data carrier classes** with minimal boilerplate code. Records are best suited for **storing immutable data** like DTOs (Data Transfer Objects).  

---

## **1️⃣ Why Use `record`?**
- **Auto-generates boilerplate code** (getters, `equals()`, `hashCode()`, `toString()`, etc.).
- **Immutable by default** (fields are `final`).
- **Concise syntax** for data models.

---

## **2️⃣ Basic Example**
```java
// Defining a record
public record Person(String name, int age) {}

public class Main {
    public static void main(String[] args) {
        Person p = new Person("Alice", 25);
        System.out.println(p.name()); // Getter for name
        System.out.println(p.age());  // Getter for age
        System.out.println(p);        // Auto-generated toString()
    }
}
```
**Output:**
```
Alice
25
Person[name=Alice, age=25]
```

✔ **Notice:**
- No need to write **constructors, getters, `toString()`, `equals()`, or `hashCode()`**.  
- Fields are **implicitly `private final`** and **cannot be modified**.

---

## **3️⃣ Custom Constructor in `record`**
Records allow **custom constructors** but must delegate to the compact constructor.

```java
public record Employee(String name, double salary) {
    // Custom constructor with validation
    public Employee {
        if (salary < 0) {
            throw new IllegalArgumentException("Salary cannot be negative!");
        }
    }
}

class Main {
    public static void main(String[] args) {
        Employee e1 = new Employee("Bob", 5000.0);
        System.out.println(e1);

        // Employee e2 = new Employee("Alice", -1000.0); // Throws Exception
    }
}
```

✔ **This constructor ensures salary is always positive.**

---

## **4️⃣ Custom Methods in `record`**
Even though records are concise, they can still **contain methods**.

```java
public record Rectangle(int width, int height) {
    public int area() {
        return width * height;
    }
}

class Main {
    public static void main(String[] args) {
        Rectangle rect = new Rectangle(5, 10);
        System.out.println("Area: " + rect.area()); // Area: 50
    }
}
```

✔ **Methods can be added inside a `record` just like a normal class.**

---

## **5️⃣ Static Fields and Methods in `record`**
```java
public record Car(String model, double price) {
    static int totalCarsSold = 0;  // Static fields are allowed

    public static void incrementSales() {
        totalCarsSold++;
    }
}

class Main {
    public static void main(String[] args) {
        Car.incrementSales();
        System.out.println("Total Cars Sold: " + Car.totalCarsSold);
    }
}
```

✔ **Static fields and methods are allowed in records.**

---

## **6️⃣ Record with Inheritance (Interfaces Only, No Class Extension)**
Records **cannot extend classes** but can **implement interfaces**.

```java
interface Vehicle {
    void drive();
}

public record Bike(String model) implements Vehicle {
    @Override
    public void drive() {
        System.out.println(model + " is being driven.");
    }
}

class Main {
    public static void main(String[] args) {
        Bike bike = new Bike("Ducati");
        bike.drive();  // Ducati is being driven.
    }
}
```

✔ **Records can implement interfaces but not extend classes.**

---

## **7️⃣ Nested Records**
Records can be **nested** inside other records or classes.

```java
public record Book(String title, Author author) {
    public record Author(String name, String nationality) {}
}

class Main {
    public static void main(String[] args) {
        Book.Author author = new Book.Author("J.K. Rowling", "British");
        Book book = new Book("Harry Potter", author);
        
        System.out.println(book);
    }
}
```

✔ **Records can be used to create hierarchical data structures.**

---

## **8️⃣ Serialization & Records**
Records are **Serializable** by default.

```java
import java.io.*;

public record Product(String name, double price) implements Serializable {}

class Main {
    public static void main(String[] args) throws IOException, ClassNotFoundException {
        Product p = new Product("Laptop", 1200.99);

        // Serialization
        ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream("product.ser"));
        out.writeObject(p);
        out.close();

        // Deserialization
        ObjectInputStream in = new ObjectInputStream(new FileInputStream("product.ser"));
        Product deserializedProduct = (Product) in.readObject();
        in.close();

        System.out.println(deserializedProduct);
    }
}
```

✔ **Records are serializable just like normal classes.**

---

## **9️⃣ When to Use `record` vs. `class`?**
| Feature | **record** | **class** |
|---------|-----------|-----------|
| **Boilerplate Code** | Less (auto-generates methods) | More (manual implementation) |
| **Immutability** | **Immutable** (all fields `final`) | Mutable (default) |
| **Extensibility** | Cannot extend other classes | Can extend other classes |
| **Use Case** | DTOs, Read-only models, Data carriers | Business logic, complex models |

✔ **Use `record` for immutable data models** (DTOs, Value Objects).  
✔ **Use `class` when mutability or inheritance is required.**

---

## **🔹 Conclusion**
- ✅ **Records simplify data modeling** by reducing boilerplate code.
- ✅ **Immutable by default**, great for DTOs, API responses, and configuration.
- ✅ **Can implement interfaces** but **cannot extend classes**.
- ✅ **Supports custom methods, static fields, and constructors**.

Would you like more advanced examples, such as **using records with Lombok or Spring Boot**? 🚀