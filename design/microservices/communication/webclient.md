Using `WebClient` with `Flux` and `Mono` in Spring WebFlux allows for **non-blocking, asynchronous HTTP communication**. It’s part of the reactive stack and offers powerful capabilities for reactive programming. Below is a comprehensive breakdown with best practices, use cases, and different scenarios.

---

## **Best Practices**

| Best Practice | Description |
|---------------|-------------|
| **Prefer `Mono` for single-value responses** | Use `Mono` for calls expecting 0 or 1 response. |
| **Use `Flux` for stream of values** | For APIs returning multiple items (e.g., JSON arrays or Server-Sent Events). |
| **Avoid `.block()` in production** | `.block()` converts async to sync, negating benefits of WebFlux. |
| **Set timeouts and retry policies** | Prevent resource leakage in case of slow/downstream services. |
| **Gracefully handle errors with `onError...` operators** | Prevent crashing pipelines on exceptions. |
| **Use connection pooling and compression** | Boost performance by reusing connections and reducing payload size. |

---

## **Key Use Cases**

1. **Calling external REST APIs**
2. **Streaming real-time data**
3. **Non-blocking communication between microservices**
4. **Asynchronous data pipelines**
5. **Reactive file or DB access with R2DBC**

---

## **Common Scenarios with Code Examples**

### **1. Basic GET (Mono)**

```java
WebClient webClient = WebClient.create("https://jsonplaceholder.typicode.com");

Mono<Post> postMono = webClient.get()
    .uri("/posts/{id}", 1)
    .retrieve()
    .bodyToMono(Post.class);

postMono.subscribe(System.out::println); // async
```

### **2. Streaming with Flux**

```java
Flux<Post> postFlux = webClient.get()
    .uri("/posts")
    .retrieve()
    .bodyToFlux(Post.class);

postFlux.subscribe(post -> System.out.println(post.getTitle()));
```

### **3. POST with JSON body**

```java
Post post = new Post("New Title", "New Content");

Mono<Post> response = webClient.post()
    .uri("/posts")
    .contentType(MediaType.APPLICATION_JSON)
    .bodyValue(post)
    .retrieve()
    .bodyToMono(Post.class);
```

### **4. Handling Errors Gracefully**

```java
Mono<Post> safeCall = webClient.get()
    .uri("/invalid-endpoint")
    .retrieve()
    .onStatus(HttpStatus::is4xxClientError, clientResponse -> 
        Mono.error(new RuntimeException("Client error"))
    )
    .bodyToMono(Post.class)
    .onErrorResume(e -> Mono.empty());
```

### **5. Retry on Failure**

```java
Mono<Post> retryCall = webClient.get()
    .uri("/posts/1")
    .retrieve()
    .bodyToMono(Post.class)
    .retryWhen(Retry.backoff(3, Duration.ofSeconds(1)));
```

### **6. Using `.block()` in Synchronous Context (Not recommended for reactive chains)**

```java
Post result = webClient.get()
    .uri("/posts/1")
    .retrieve()
    .bodyToMono(Post.class)
    .block(); // blocking
```

---

## **Summary of Mono vs Flux**

| Use Case | Mono | Flux |
|----------|------|------|
| Single value (or empty) | `Mono<T>` | Not suitable |
| Multiple values | Not suitable | `Flux<T>` |
| HTTP GET (single JSON object) | Yes | No |
| HTTP GET (array/stream) | No | Yes |

---

Would you like a Spring Boot project using WebClient, Mono, Flux with tests and logging? I can generate it or walk you through it.


Great! Let's walk through the best practices and use cases for using Spring WebClient with Flux and Mono in both synchronous and asynchronous communication. I'll include code examples for each scenario.

---

### **1. Basic Setup of WebClient**

```java
@Bean
public WebClient webClient(WebClient.Builder builder) {
    return builder.baseUrl("https://api.example.com").build();
}
```

---

### **2. Asynchronous GET Request (using `Mono`)**

```java
public Mono<String> fetchMessage() {
    return webClient.get()
        .uri("/message")
        .retrieve()
        .bodyToMono(String.class);
}
```

### **Best Practices**
- Use `retry` for transient errors.
- Use `.timeout(Duration.ofSeconds(x))` to prevent hanging.

---

### **3. Asynchronous POST Request (with request body and custom headers)**

```java
public Mono<ResponseEntity<String>> postData(MyRequestBody body) {
    return webClient.post()
        .uri("/submit")
        .header(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_JSON_VALUE)
        .bodyValue(body)
        .retrieve()
        .toEntity(String.class);
}
```

---

### **4. Streaming Data with `Flux`**

```java
public Flux<MyEvent> streamEvents() {
    return webClient.get()
        .uri("/events")
        .accept(MediaType.TEXT_EVENT_STREAM)
        .retrieve()
        .bodyToFlux(MyEvent.class);
}
```

**Use Case:** Event-driven architecture, SSE (Server-Sent Events).

---

### **5. Synchronous (Blocking) Call - Not Recommended for Reactive Pipelines**

```java
public String fetchDataSync() {
    return webClient.get()
        .uri("/data")
        .retrieve()
        .bodyToMono(String.class)
        .block(); // blocking
}
```

**Note:** Only use `block()` when necessary and not within a reactive flow.

---

### **6. Exception Handling**

```java
public Mono<String> getWithErrorHandling() {
    return webClient.get()
        .uri("/error-prone")
        .retrieve()
        .onStatus(HttpStatus::is4xxClientError, response -> 
            Mono.error(new RuntimeException("Client error")))
        .bodyToMono(String.class)
        .onErrorResume(e -> Mono.just("Fallback response"));
}
```

---

### **7. Logging**

```yaml
# application.yml
logging:
  level:
    org.springframework.web.reactive.function.client.ExchangeFunctions: TRACE
```

This will log the request and response for debugging.

---

### **8. Unit Test with `WebClient` using MockWebServer**

```java
@Test
void testGetMessage() throws Exception {
    mockWebServer.enqueue(new MockResponse()
        .setBody("Hello")
        .addHeader("Content-Type", "text/plain"));

    String response = webClient.get()
        .uri("/message")
        .retrieve()
        .bodyToMono(String.class)
        .block();

    assertEquals("Hello", response);
}
```

---

Let me know if you'd like:
- Integration with Spring Boot services or controllers
- Retry/backoff strategies
- Circuit breaker pattern with Resilience4j
- TestContainer-based integration tests with WebClient

Want an actual example project setup too?

Using `WebClient` with `Flux` and `Mono` in Spring WebFlux allows for **non-blocking, asynchronous HTTP communication**. It’s part of the reactive stack and offers powerful capabilities for reactive programming. Below is a comprehensive breakdown with best practices, use cases, and different scenarios.

---

## **Best Practices**

| Best Practice | Description |
|---------------|-------------|
| **Prefer `Mono` for single-value responses** | Use `Mono` for calls expecting 0 or 1 response. |
| **Use `Flux` for stream of values** | For APIs returning multiple items (e.g., JSON arrays or Server-Sent Events). |
| **Avoid `.block()` in production** | `.block()` converts async to sync, negating benefits of WebFlux. |
| **Set timeouts and retry policies** | Prevent resource leakage in case of slow/downstream services. |
| **Gracefully handle errors with `onError...` operators** | Prevent crashing pipelines on exceptions. |
| **Use connection pooling and compression** | Boost performance by reusing connections and reducing payload size. |

---

## **Key Use Cases**

1. **Calling external REST APIs**
2. **Streaming real-time data**
3. **Non-blocking communication between microservices**
4. **Asynchronous data pipelines**
5. **Reactive file or DB access with R2DBC**

---

## **Common Scenarios with Code Examples**

### **1. Basic GET (Mono)**

```java
WebClient webClient = WebClient.create("https://jsonplaceholder.typicode.com");

Mono<Post> postMono = webClient.get()
    .uri("/posts/{id}", 1)
    .retrieve()
    .bodyToMono(Post.class);

postMono.subscribe(System.out::println); // async
```

### **2. Streaming with Flux**

```java
Flux<Post> postFlux = webClient.get()
    .uri("/posts")
    .retrieve()
    .bodyToFlux(Post.class);

postFlux.subscribe(post -> System.out.println(post.getTitle()));
```

### **3. POST with JSON body**

```java
Post post = new Post("New Title", "New Content");

Mono<Post> response = webClient.post()
    .uri("/posts")
    .contentType(MediaType.APPLICATION_JSON)
    .bodyValue(post)
    .retrieve()
    .bodyToMono(Post.class);
```

### **4. Handling Errors Gracefully**

```java
Mono<Post> safeCall = webClient.get()
    .uri("/invalid-endpoint")
    .retrieve()
    .onStatus(HttpStatus::is4xxClientError, clientResponse -> 
        Mono.error(new RuntimeException("Client error"))
    )
    .bodyToMono(Post.class)
    .onErrorResume(e -> Mono.empty());
```

### **5. Retry on Failure**

```java
Mono<Post> retryCall = webClient.get()
    .uri("/posts/1")
    .retrieve()
    .bodyToMono(Post.class)
    .retryWhen(Retry.backoff(3, Duration.ofSeconds(1)));
```

### **6. Using `.block()` in Synchronous Context (Not recommended for reactive chains)**

```java
Post result = webClient.get()
    .uri("/posts/1")
    .retrieve()
    .bodyToMono(Post.class)
    .block(); // blocking
```

---

## **Summary of Mono vs Flux**

| Use Case | Mono | Flux |
|----------|------|------|
| Single value (or empty) | `Mono<T>` | Not suitable |
| Multiple values | Not suitable | `Flux<T>` |
| HTTP GET (single JSON object) | Yes | No |
| HTTP GET (array/stream) | No | Yes |

---

Would you like a Spring Boot project using WebClient, Mono, Flux with tests and logging? I can generate it or walk you through it.