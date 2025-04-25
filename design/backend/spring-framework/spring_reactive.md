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