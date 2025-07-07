Here’s a complete guide to the **core Object-Oriented Design (OOD) concepts** in Java, with simple explanations and Java code examples.

---

## ✅ 1. **Class and Object**

### 📌 Definition:

* **Class** is a blueprint.
* **Object** is an instance of a class.

### ✅ Example:

```java
class Car {
    String brand = "Toyota";
}

public class Main {
    public static void main(String[] args) {
        Car car = new Car();  // object
        System.out.println(car.brand);
    }
}
```

---

## ✅ 2. **Encapsulation**

### 📌 Definition:

* Wrapping data and methods in one unit.
* Use `private` fields with `public` getters/setters.

### ✅ Example:

```java
class Person {
    private String name;

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }
}
```

---

## ✅ 3. **Abstraction**

### 📌 Definition:

* Hiding internal implementation and showing only functionality.
* Achieved using `abstract` classes or `interfaces`.

### ✅ Example:

```java
abstract class Animal {
    abstract void makeSound();  // abstract method
}

class Dog extends Animal {
    void makeSound() {
        System.out.println("Woof");
    }
}
```

---

## ✅ 4. **Inheritance**

### 📌 Definition:

* One class inherits properties and behaviors from another.

### ✅ Example:

```java
class Animal {
    void eat() {
        System.out.println("Eating");
    }
}

class Dog extends Animal {
    void bark() {
        System.out.println("Barking");
    }
}
```

---

## ✅ 5. **Polymorphism**

### 📌 Definition:

* One interface, many implementations.
* Two types: Compile-time (method overloading), Runtime (method overriding)

### ✅ Example (Runtime Polymorphism):

```java
class Animal {
    void makeSound() {
        System.out.println("Some sound");
    }
}

class Cat extends Animal {
    void makeSound() {
        System.out.println("Meow");
    }
}
```

---

## ✅ 6. **Association**

### 📌 Definition:

* General relationship between two classes.
* Example: A `Teacher` teaches `Student`.

### ✅ Example:

```java
class Teacher {
    String name;
}

class Student {
    String name;
    Teacher teacher;
}
```

---

## ✅ 7. **Aggregation**

### 📌 Definition:

* A "has-a" relationship.
* Child can exist independently of the parent.

### ✅ Example:

```java
class Department {
    String name;
}

class School {
    List<Department> departments; // aggregation
}
```

---

## ✅ 8. **Composition**

### 📌 Definition:

* Stronger form of aggregation.
* Child cannot exist without parent.

### ✅ Example:

```java
class Heart {
    void beat() {
        System.out.println("Beating");
    }
}

class Human {
    private final Heart heart = new Heart(); // composition

    void live() {
        heart.beat();
    }
}
```

---

## ✅ 9. **Dependency**

### 📌 Definition:

* One class depends on another to perform its function.

### ✅ Example:

```java
class Engine {
    void start() {
        System.out.println("Engine starting...");
    }
}

class Car {
    private Engine engine;

    Car(Engine engine) {
        this.engine = engine;
    }

    void drive() {
        engine.start();
    }
}
```

---

## ✅ Summary Table

| Concept       | Definition                         | Example Keyword / Form      |
| ------------- | ---------------------------------- | --------------------------- |
| Class/Object  | Blueprint / Instance               | `new`, `class`              |
| Encapsulation | Hide data with accessors           | `private`, `get/set`        |
| Abstraction   | Hide internal details              | `abstract`, `interface`     |
| Inheritance   | Extend functionality               | `extends`                   |
| Polymorphism  | Many forms (override/overload)     | `@Override`                 |
| Association   | General relation                   | Object references           |
| Aggregation   | Has-a (can exist independently)    | Contained object reference  |
| Composition   | Has-a (cannot exist independently) | Strong ownership            |
| Dependency    | One uses another                   | Constructor or method param |

---

Let me know if you'd like UML diagrams or real-world examples for each.
