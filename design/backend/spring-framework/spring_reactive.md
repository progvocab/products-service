# **🔹 Asynchronous API Calls in Spring Using Reactive Programming**  

Spring provides **Reactive Programming** support via **Project Reactor**, enabling non-blocking, asynchronous data processing. The core components for building reactive APIs in Spring Boot are:  
- **Flux & Mono** (Reactive Data Types)  
- **WebClient** (Non-blocking HTTP client)  
- **Spring WebFlux** (Alternative to Spring MVC for reactive APIs)  

---

# **🔹 Key Components of Reactive Programming**
| Component | Description |
|-----------|------------|
| **Mono<T>** | Represents a **single** asynchronous result (0 or 1 item). |
| **Flux<T>** | Represents a **stream** of asynchronous results (0 to N items). |
| **WebClient** | Non-blocking, reactive alternative to `RestTemplate` for API calls. |
| **Schedulers** | Controls execution threading (e.g., `Schedulers.parallel()`). |
| **StepVerifier** | Used for testing reactive streams. |

---

# **🔹 1. Mono & Flux (Reactive Data Types)**
📌 **Mono** → Use when expecting **one** response (e.g., fetching a single user).  
📌 **Flux** → Use when expecting **multiple** responses (e.g., fetching a list of users).  

### **✅ Example: Using `Mono` and `Flux`**
```java
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

public class ReactiveExample {
    public static void main(String[] args) {
        // Mono Example (0 or 1 item)
        Mono<String> monoExample = Mono.just("Hello, Reactive World!");
        monoExample.subscribe(System.out::println);

        // Flux Example (0 to N items)
        Flux<String> fluxExample = Flux.just("Spring", "WebFlux", "Reactor");
        fluxExample.subscribe(System.out::println);
    }
}
```
🔹 **Mono emits one item, while Flux emits a stream of items!**  

---

# **🔹 2. WebClient (Non-blocking API Calls)**
📌 **`WebClient` is the replacement for `RestTemplate`** in reactive programming.  
📌 It makes **asynchronous, non-blocking** HTTP calls.  

### **✅ Example: Calling an External API Using WebClient**
```java
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

public class WebClientExample {
    public static void main(String[] args) {
        WebClient webClient = WebClient.create("https://jsonplaceholder.typicode.com");

        // Making a GET request to fetch a user
        Mono<String> userMono = webClient.get()
                .uri("/users/1")
                .retrieve()
                .bodyToMono(String.class);

        userMono.subscribe(System.out::println); // Asynchronous call
    }
}
```
🔹 **WebClient automatically handles asynchronous requests and responses!** 🚀  

---

# **🔹 3. Spring WebFlux (Reactive Controller)**
📌 **Spring WebFlux is the reactive alternative to Spring MVC.**  
📌 **Controllers return `Mono` or `Flux` instead of regular objects.**  

### **✅ Example: Reactive REST API with WebFlux**
```java
import org.springframework.web.bind.annotation.*;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.time.Duration;
import java.util.List;

@RestController
@RequestMapping("/users")
public class UserController {

    // Return a single user asynchronously (Mono)
    @GetMapping("/{id}")
    public Mono<String> getUser(@PathVariable String id) {
        return Mono.just("User-" + id);
    }

    // Return a stream of users asynchronously (Flux)
    @GetMapping
    public Flux<String> getAllUsers() {
        return Flux.fromIterable(List.of("Alice", "Bob", "Charlie"))
                   .delayElements(Duration.ofSeconds(1)); // Simulate delay
    }
}
```
🔹 **Calls to these endpoints are fully reactive!** 🚀  

---

# **🔹 4. WebClient for Asynchronous API Calls Between Microservices**
📌 **Use `WebClient` to call another microservice in a non-blocking way.**  

### **✅ Example: Calling Another Microservice**
```java
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

@Service
public class UserService {
    private final WebClient webClient;

    public UserService(WebClient.Builder webClientBuilder) {
        this.webClient = webClientBuilder.baseUrl("http://user-service").build();
    }

    public Mono<String> getUserById(String userId) {
        return webClient.get()
                .uri("/users/" + userId)
                .retrieve()
                .bodyToMono(String.class);
    }
}
```
🔹 **Now, `UserService` makes an asynchronous API call!**  

---

# **🔹 5. Using WebClient with Parallel Calls**
📌 **Parallel API calls improve performance.**  

### **✅ Example: Calling Two APIs in Parallel**
```java
public Mono<String> fetchUserAndOrders(String userId) {
    Mono<String> userMono = webClient.get().uri("/users/" + userId)
                                     .retrieve().bodyToMono(String.class);

    Mono<String> ordersMono = webClient.get().uri("/orders/" + userId)
                                       .retrieve().bodyToMono(String.class);

    return Mono.zip(userMono, ordersMono, (user, orders) -> user + " " + orders);
}
```
🔹 **Both API calls happen in parallel, reducing response time!** 🚀  

---

# **🔹 6. Error Handling in WebClient**
📌 **Handle errors using `.onErrorResume()` or `.onStatus()`.**  

### **✅ Example: Error Handling in API Calls**
```java
public Mono<String> getUserWithErrorHandling(String userId) {
    return webClient.get()
            .uri("/users/" + userId)
            .retrieve()
            .onStatus(status -> status.is4xxClientError(), response -> Mono.error(new RuntimeException("User not found")))
            .bodyToMono(String.class)
            .onErrorResume(e -> Mono.just("Fallback User"));
}
```
🔹 **If the user is not found, it returns a fallback response!**  

---

# **🔹 7. Testing WebFlux with StepVerifier**
📌 **Use `StepVerifier` to test reactive APIs.**  

### **✅ Example: Unit Test for Reactive API**
```java
import org.junit.jupiter.api.Test;
import reactor.core.publisher.Mono;
import reactor.test.StepVerifier;

public class ReactiveTest {

    @Test
    void testMono() {
        Mono<String> mono = Mono.just("Spring WebFlux");

        StepVerifier.create(mono)
                    .expectNext("Spring WebFlux")
                    .verifyComplete();
    }
}
```
🔹 **Ensures reactive logic works as expected!** ✅  

---

# **🔹 Summary**
| Feature | Description |
|---------|------------|
| **Mono** | Represents **0 or 1** item. |
| **Flux** | Represents **0 to N** items (stream). |
| **WebClient** | Non-blocking API calls (alternative to RestTemplate). |
| **Spring WebFlux** | Reactive alternative to Spring MVC. |
| **Schedulers** | Controls execution threads (parallel, elastic, bounded). |
| **StepVerifier** | Used for testing reactive streams. |

---

# **🚀 Do you need a full project demo on Spring WebFlux?**


**Spring WebFlux** to **push updates whenever the Employee table data changes** (like if a new employee is added, or salary is updated).

This is a **reactive push** — not pull.  
In **Spring WebFlux**, we can achieve this using a combination of:

- **Flux** (reactive stream of events)
- **Sinks** (to manually push new events)
- **Database polling** (simplest way without DB triggers)
- (Advanced) **Change Data Capture (CDC)** tools like Debezium if you want no-polling (I can show that too later if you want).

---

# Basic working idea:  
1. **Backend** keeps checking database for changes (polling or via CDC).  
2. If change detected, **push new Employee list / change event** into a **Flux** sink.  
3. Clients (Browser/Apps) subscribe and **get real-time updates**.

---

# Here’s a simple version:

### 1. Setup Maven Dependencies

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-webflux</artifactId>
</dependency>

<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-data-r2dbc</artifactId> <!-- Reactive DB -->
</dependency>

<dependency>
    <groupId>io.r2dbc</groupId>
    <artifactId>r2dbc-h2</artifactId> <!-- Example: using H2 -->
</dependency>
```

*(For Oracle reactive drivers, you can configure differently, tell me if needed.)*

---

### 2. Employee Entity (Reactive Repository)

```java
@Table("employee")
public class Employee {
    @Id
    private Long id;
    private String name;
    private Double salary;
    
    // getters and setters
}
```

```java
@Repository
public interface EmployeeRepository extends ReactiveCrudRepository<Employee, Long> {
}
```

---

### 3. Create a **Flux Sink** to Push Updates

```java
@Service
public class EmployeeChangePublisher {

    private final Sinks.Many<List<Employee>> sink;

    @Autowired
    private EmployeeRepository employeeRepository;

    public EmployeeChangePublisher() {
        this.sink = Sinks.many().replay().latest(); // cache latest value
    }

    public Flux<List<Employee>> getEmployeeChanges() {
        return sink.asFlux();
    }

    public void publishChanges() {
        employeeRepository.findAll()
            .collectList()
            .subscribe(employees -> sink.tryEmitNext(employees));
    }
}
```

- **Sinks.many().replay().latest()**: Keeps the latest state.
- **publishChanges()**: Pulls all employees and pushes to the sink.

---

### 4. Polling Database Periodically

```java
@Component
public class EmployeePoller {

    @Autowired
    private EmployeeChangePublisher publisher;

    @Scheduled(fixedDelay = 5000) // every 5 seconds
    public void pollDatabase() {
        publisher.publishChanges();
    }
}
```

_(Enable scheduling with `@EnableScheduling` on Spring Boot app class)_

---

### 5. Create a WebFlux Controller to Serve Updates

```java
@RestController
@RequestMapping("/employees")
public class EmployeeController {

    @Autowired
    private EmployeeChangePublisher publisher;

    @GetMapping(value = "/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public Flux<List<Employee>> streamEmployees() {
        return publisher.getEmployeeChanges();
    }
}
```

- **`TEXT_EVENT_STREAM_VALUE`** enables **Server-Sent Events (SSE)**.
- Clients can subscribe and get **continuous updates**!

---

### 6. Client Example (Javascript)

```javascript
const eventSource = new EventSource('http://localhost:8080/employees/stream');

eventSource.onmessage = function(event) {
    console.log('Received update:', JSON.parse(event.data));
};
```

---

# **Summary**

| Part | Purpose |
|:-----|:--------|
| `EmployeeChangePublisher` | Manages pushing employee updates |
| `EmployeePoller` | Polls DB and triggers updates |
| `EmployeeController` | Exposes the Flux via WebFlux endpoint |
| Client (JS) | Subscribes using EventSource |

---

# **Pros and Cons of Polling Approach**

| Pros | Cons |
|:-----|:-----|
| Easy to build | Small delay (5s, 10s etc.) |
| No DB setup change | Unnecessary queries if no data change |

---

# **Advanced Alternative**  
**If you want absolutely real-time** without polling:
- Use **Debezium** with Kafka (captures DB changes instantly)
- Spring Boot app listens Kafka and updates Flux  
_(Let me know if you want to see this version too!)_

---

# **Would you like me to show:**
- Pushing **only updated Employee** instead of full list?
- Using **Database triggers** instead of polling?
- Or **Debezium CDC real-time setup**?

Type:  
> "Show updated-only version" / "Show Debezium" / "Show triggers approach"  
and I’ll continue!  
---
Would you also want this integrated with **Keycloak secured API**? (real-time + secured updates!)



**Good question!**  
The answer is: **Yes, conceptually similar — but some important differences too.**

Let’s break it down properly:

---

# 1. **Angular Observable + Firebase Realtime Updates**

- **Firebase** backend (like Realtime Database / Firestore) automatically **pushes changes** to the frontend.
- In Angular, you usually write:

```typescript
this.db.list('employees').valueChanges().subscribe(data => {
  console.log(data);
});
```

- Here, Firebase itself **manages**:
  - Listening to backend changes
  - Pushing real-time updates
  - WebSocket or proprietary protocols under the hood
- You don’t **poll** in Firebase — **you truly listen**.

Firebase behaves almost like a **Reactive Database** natively.

---

# 2. **Spring WebFlux + Sinks + Scheduled Polling**

- Spring Boot with WebFlux is **making a normal database reactive** by manually pushing changes via `Sinks` and `Flux`.
- **Without CDC tools (like Debezium)**, you **poll** periodically and push changes.
- It **simulates real-time**.

If you use **Debezium** or **Database Triggers**, you can make it almost **exactly like Firebase** — truly event-based.

---

# 3. **Comparison Table**

| Feature | Angular + Firebase | Spring Boot + WebFlux (Polling) |
|:--------|:-------------------|:-------------------------------|
| Push real-time updates | Native | Manual (polling/sinks) |
| Backend event system | Built-in | You build it (or use Debezium) |
| Communication | WebSocket / long-poll | Server-Sent Events (SSE) |
| Reliability | Highly consistent | Depends on polling delay |
| Database | NoSQL (Firestore) | Relational/NoSQL |
| Complexity | Simple to code | Needs setup (sinks, events) |

---

# 4. **Where they are similar**
- Both **stream data changes** reactively.
- Both **clients subscribe** to backend updates.
- Both use **observable patterns**.

---

# 5. **Where they are different**
- In Firebase, database **itself** supports real-time pub/sub.
- In Spring Boot, **we** are creating a **reactive layer** on top of a traditional database.

Thus:
- **Firebase Observable** = Database is publisher.
- **Spring WebFlux** = We make an **application layer publisher**.

---

# 6. **Simple Diagram**

```
Angular <--- Observable subscription <--- Firebase (auto pushes changes)

Angular <--- SSE (Flux) <--- Spring Boot (manual or Debezium) <--- Database
```

---

# 7. **Design Patterns involved**

| Pattern | Description |
|:--------|:------------|
| Observer | Clients react to data changes. |
| Publisher-Subscriber | Backend emits changes; clients listen. |
| Reactor pattern (in Spring) | Non-blocking event-driven flow. |

---

# 8. **Important Note**  
If you integrate **Debezium** or **Database Triggers**, your Spring Boot setup will become truly **event-based** like Firebase.

Polling every few seconds is a **simple approximation** if you don't want to set up CDC.

---

# **In short:**  
**Spring WebFlux polling approach is functionally similar to Angular + Firebase Observable, but Firebase is natively push-based while Spring Boot WebFlux needs to simulate or integrate event-driven behavior manually.**

---
Would you like me to show:

- How to upgrade this with **Debezium** to make it fully real-time event-driven?
- A simple **database trigger** based solution too?

(very practical if you are building a high-frequency real-time system!)  
Let me know!  
We can level up this architecture together if you want!