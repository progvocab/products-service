Great observation — the usage of `WebClient`, `Flux`, and `Mono` in **Spring WebFlux** follows a combination of modern **design patterns** and **reactive programming paradigms**. Let's break it down:

---

### **Design Patterns Behind WebClient, Flux, and Mono**

#### 1. **Builder Pattern** – `WebClient`

Instead of passing all values to a constructor or method, you build the request step by step:

```java
WebClient.create()
         .get()
         .uri("/example")
         .header("Authorization", "Bearer token")
         .retrieve()
         .bodyToMono(String.class);
```

This is **Builder Pattern**:
- It allows constructing complex objects step-by-step.
- Each method returns the current builder instance.
- Makes the code more readable and chainable.

---

#### 2. **Chain of Responsibility / Fluent Interface**

- The chaining (`.get().uri().retrieve()`) is an example of **Fluent Interface**.
- It makes the code expressive and readable, encouraging immutability.

---

#### 3. **Reactor Pattern** – `Flux` and `Mono`

- Spring WebFlux is built on **Project Reactor** which implements the **Reactive Streams specification**.
- The core of `Mono` and `Flux` is **asynchronous, non-blocking event handling**, which is the essence of the **Reactor Pattern**.

---

#### 4. **Publisher-Subscriber Pattern** – Flux & Mono

```java
Mono<String> result = webClient.get().uri("/data").retrieve().bodyToMono(String.class);
result.subscribe(System.out::println);
```

This is a **Publisher-Subscriber** or **Observer** pattern:
- `Mono`/`Flux` are Publishers.
- You can `subscribe()` to them to receive data when it's ready.

---

### **Why Not One Method With All Params?**

Using method chaining instead of one giant method:
- Makes the code **declarative and composable**.
- Supports **optional and dynamic** configuration easily.
- Reduces parameter complexity and enhances clarity.

---

### Summary Table

| Component   | Pattern(s) Used                     | Purpose                          |
|-------------|-------------------------------------|-----------------------------------|
| `WebClient` | Builder, Fluent Interface           | Build HTTP requests fluently     |
| `Mono/Flux` | Reactor, Publisher-Subscriber       | Non-blocking async programming   |
| WebClient call chain | Chain of Responsibility, Builder | Step-by-step configuration       |

---

Let me know if you'd like a visual UML or example transforming this into a custom builder class.