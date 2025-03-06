Yes, in addition to the **Gang of Four (GoF) creational design patterns**, other creational patterns exist, often used in specific contexts like concurrency, high-performance computing, or domain-driven design. Below is a **comprehensive list** of creational design patterns:

---

## **1. Gang of Four (GoF) Creational Patterns**  
These are the **most widely recognized creational patterns** from the GoF book:

| **Pattern**              | **Description** |
|--------------------------|---------------|
| **Singleton**            | Ensures only one instance of a class exists. |
| **Factory Method**       | Delegates instantiation to subclasses. |
| **Abstract Factory**     | Creates families of related objects. |
| **Builder**             | Constructs complex objects step by step. |
| **Prototype**           | Creates objects by cloning existing instances. |

---

## **2. Extended Creational Patterns**  
These are additional patterns that are variations or enhancements of GoF patterns:

| **Pattern**              | **Description** |
|--------------------------|---------------|
| **Object Pool**          | Maintains a pool of reusable objects to optimize performance. |
| **Lazy Initialization**  | Delays object creation until it is actually needed. |
| **Multiton**            | Like Singleton but allows multiple named instances. |
| **Dependency Injection** | Injects dependencies into objects instead of creating them internally. |

---

## **3. Concurrency-Specific Creational Patterns**  
These patterns are useful for **multithreading and parallel computing**:

| **Pattern**              | **Description** |
|--------------------------|---------------|
| **Thread-Safe Singleton** | Ensures Singleton works safely in multi-threaded applications. |
| **Resource Acquisition Is Initialization (RAII)** | Ensures resource allocation and release happen within object lifecycle. |
| **Factory Pool**         | Similar to Object Pool but optimized for concurrent environments. |

---

## **4. Domain-Driven Design (DDD) Creational Patterns**  
These are commonly used in **enterprise applications and domain-driven design**:

| **Pattern**              | **Description** |
|--------------------------|---------------|
| **Aggregate Factory**    | Creates an entire aggregate (group of related objects) as a single unit. |
| **Service Locator**      | Acts as a registry for creating and accessing dependencies. |
| **Prototype Registry**   | Manages multiple prototype objects for cloning. |

---

## **5. Cloud & Microservices Creational Patterns**  
These are used in **cloud computing, containerized environments, and microservices**:

| **Pattern**              | **Description** |
|--------------------------|---------------|
| **Self-Registering Factory** | Automatically registers new instances of services. |
| **Configuration-Driven Factory** | Uses external configurations to decide object creation. |
| **Service Bootstrap** | Initializes services dynamically based on runtime conditions. |

---

### **Final Summary**
The **most common** creational patterns are the GoF ones, but many others exist, including **multithreading, DDD, and cloud-native patterns**.  

Would you like **detailed examples** for any specific pattern?