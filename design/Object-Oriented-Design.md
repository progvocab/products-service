Here’s a complete guide to the **core Object-Oriented Design (OOD) concepts** in Java, with simple explanations and Java code examples.



### **Class and Object**



* **Class** is a blueprint.
* **Object** is an instance of a class.



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



### **Encapsulation**



* Wrapping data and methods in one unit.
* Use `private` fields with `public` getters/setters.



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



### **Abstraction**



* Hiding internal implementation and showing only functionality.
* Achieved using `abstract` classes or `interfaces`.



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



### **Inheritance**



* One class inherits properties and behaviors from another.



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



### **Polymorphism**



* One interface, many implementations.
* Two types: Compile-time (method overloading), Runtime (method overriding)

#### Runtime Polymorphism

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

 

###   **Association**


* Independent Lifecycle
* General relationship ("uses a" / "knows a") between two classes.
* Unidirectional or Bidirectional
* Example: A `Teacher` teaches `Student`.
* No Whole Part relationship


```java
class Teacher {
    String name;
}

class Student {
    String name;
    Teacher teacher;
}
```



### **Aggregation**



* A "has-a" relationship.
* Child can exist independently of the parent.
* Whole Part relationship.


```java
class Department {
    String name;
}

class School {
    List<Department> departments; // aggregation
}
```



### **Composition**


* A "has-a" relationship.
* Stronger form of aggregation.
* Child cannot exist without parent.



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



### **Dependency**



* One class depends on another to perform its function.



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

 

| Concept       | Definition                         | Example Keyword / Form      |
| - | - |  |
| Class/Object  | Blueprint / Instance               | `new`, `class`              |
| Encapsulation | Hide data with accessors           | `private`, `get/set`        |
| Abstraction   | Hide internal details              | `abstract`, `interface`     |
| Inheritance   | Extend functionality               | `extends`                   |
| Polymorphism  | Many forms (override/overload)     | `@Override`                 |
| Association   | General relation                   | Object references           |
| Aggregation   | Has-a (can exist independently)    | Contained object reference  |
| Composition   | Has-a (cannot exist independently) | Strong ownership            |
| Dependency    | One uses another                   | Constructor or method param |


 
