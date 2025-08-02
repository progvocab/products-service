### ⚛️ Reactive Programming Paradigm

**Reactive Programming** is a programming paradigm focused on **asynchronous data streams** and the **propagation of change**.

Rather than asking for data or pulling values, reactive systems **react to data** as it arrives — much like an event-driven model, but with a strong emphasis on **composition**, **backpressure**, and **non-blocking execution**.

---

## 🔁 Traditional vs Reactive Example

### Traditional (Imperative):

```java
String data = getData();
System.out.println(data);
```

### Reactive (Declarative):

```java
Mono<String> data = getDataAsync();
data.subscribe(System.out::println);
```

Here, `Mono` is a **reactive stream** that will eventually emit a single value or error.

---

## 🧠 Core Principles (Reactive Manifesto)

| Principle          | Description                                     |
| ------------------ | ----------------------------------------------- |
| **Responsive**     | Systems respond in a timely manner              |
| **Resilient**      | Systems stay responsive under failure           |
| **Elastic**        | Systems stay responsive under varying load      |
| **Message Driven** | Components interact using async message passing |

---

## 📚 Key Reactive Concepts

| Concept                    | Description                                        |
| -------------------------- | -------------------------------------------------- |
| **Stream**                 | A sequence of events/data emitted over time        |
| **Observable / Publisher** | Emits data                                         |
| **Observer / Subscriber**  | Consumes data                                      |
| **Operators**              | Transform, filter, merge, buffer, etc.             |
| **Backpressure**           | Mechanism to handle overwhelming data rates        |
| **Schedulers**             | Define threading (like `subscribeOn`, `publishOn`) |

---

## 🔧 Popular Libraries

| Language   | Library                      |
| ---------- | ---------------------------- |
| Java       | Project Reactor, RxJava      |
| JavaScript | RxJS                         |
| Kotlin     | Flow, Coroutine              |
| Python     | RxPY                         |
| C#         | Reactive Extensions (Rx.NET) |

---

## 🚀 Java (Reactor) Example

```java
Flux<Integer> flux = Flux.range(1, 5)
    .map(i -> i * 2)
    .filter(i -> i > 5);

flux.subscribe(System.out::println);
```

**Output**:

```
6
8
10
```

---

## 🧱 Reactive Types in Project Reactor (Spring WebFlux)

| Type      | Description        |
| --------- | ------------------ |
| `Mono<T>` | Emits 0 or 1 item  |
| `Flux<T>` | Emits 0 to N items |

Used in `Spring WebFlux` for building **non-blocking**, event-driven REST APIs.

---

## 📦 Spring WebFlux Endpoint Example

```java
@GetMapping("/greet")
public Mono<String> greet() {
    return Mono.just("Hello, Reactive World!");
}
```

---

## ✅ Benefits of Reactive Programming

* Efficient resource usage (non-blocking)
* Handles high concurrency with fewer threads
* Better for event-driven, data-intensive apps
* Enables **stream processing** and **real-time systems**

---

## ⚠️ Challenges

* Steep learning curve
* Debugging and stack traces can be tricky
* Composition is powerful but requires care
* Not suitable for simple, synchronous tasks

---

Would you like an example comparing a Spring MVC vs WebFlux controller side-by-side, or how to write reactive services for MongoDB, Redis, or Kafka?
