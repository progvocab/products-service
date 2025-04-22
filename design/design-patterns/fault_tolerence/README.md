Sure! **Fault tolerance** is a system's ability to **continue operating** properly in the event of partial system failures. Fault tolerance design patterns are architectural and programming strategies that ensure applications **recover gracefully** or **degrade gracefully** instead of crashing.

Here’s a categorized explanation of **Fault Tolerance Design Patterns**:

---

## **1. Core Fault Tolerance Patterns**

| Pattern              | Description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| **Retry**            | Automatically retries failed operations. Useful when failures are temporary.|
| **Circuit Breaker**  | Opens the circuit after a threshold of failures to avoid overwhelming the system. |
| **Timeout**          | Prevents hanging by failing calls that exceed a time limit.                 |
| **Fallback**         | Provides an alternative response when a request fails.                     |
| **Failover**         | Switches to a backup system/component in case of failure.                   |
| **Load Balancing**   | Distributes workload across multiple nodes to avoid overloading one.        |
| **Bulkhead**         | Isolates parts of the system to prevent failures from propagating.          |
| **Graceful Degradation** | Reduces functionality while still providing basic service.              |
| **Redundancy**       | Duplicates components for availability and reliability.                    |
| **Health Monitoring**| Continuously checks system components and triggers recovery if needed.     |

---

## **2. Supporting Patterns**

| Pattern              | Description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| **Idempotency**      | Ensures repeated execution has the same effect, important for retries.      |
| **Caching**          | Serves stored results when live services fail or are slow.                  |
| **Rate Limiting**    | Prevents resource exhaustion by capping usage.                              |
| **Shadowing**        | Runs a copy of the system to test changes under real traffic without impact.|
| **Watchdog**         | Monitors services and restarts them if they fail or hang.                   |

---

## **3. Classification by Purpose**

| Category                   | Patterns                                              |
|----------------------------|-------------------------------------------------------|
| **Error Recovery**         | Retry, Fallback, Timeout, Idempotency                 |
| **Error Prevention**       | Bulkhead, Circuit Breaker, Rate Limiting              |
| **Service Continuity**     | Failover, Graceful Degradation, Redundancy, Caching   |
| **Health & Monitoring**    | Watchdog, Health Check, Shadowing                     |

---

## **Example in Spring Boot (with Resilience4j)**

```java
@Retry(name = "userService", fallbackMethod = "fallback")
@CircuitBreaker(name = "userService", fallbackMethod = "fallback")
public User getUserById(String id) {
    return restTemplate.getForObject("http://remote-service/users/" + id, User.class);
}

public User fallback(String id, Throwable e) {
    return new User(id, "Fallback User");
}
```

---

## **Relation to GoF or Architecture Patterns**

These are not part of the **Gang of Four (GoF)** patterns, but they are **system-level architectural patterns** under:

- **Resiliency**
- **High Availability**
- **Distributed Systems Architecture**

---

Would you like a mind map or a visual diagram showing how these patterns interrelate in a fault-tolerant architecture (like microservices)?