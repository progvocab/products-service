Sure! **Resiliency design patterns** are architectural and code-level strategies that ensure an application continues functioning correctly even in the face of **partial failures**, such as network issues, service downtime, or high load.

Here’s a list of key **Resiliency Patterns**, with brief explanations and use cases:

---

## **Resiliency Design Patterns**

| Pattern                      | Description                                                                 | Common Use Case                                      |
|------------------------------|-----------------------------------------------------------------------------|------------------------------------------------------|
| **Retry**                    | Automatically retries a failed operation a certain number of times.         | Temporary network issues, slow services              |
| **Circuit Breaker**          | Stops calling a failing service temporarily to prevent overload and allow recovery. | Unavailable or unstable downstream services         |
| **Timeout**                  | Limits how long a request waits for a response before failing.              | Prevents resource blocking from unresponsive services|
| **Bulkhead**                 | Isolates components so failure in one doesn't bring down others.           | Thread pools, service isolation                      |
| **Fallback**                 | Provides a default behavior when a service fails.                          | Return cached/default response if live service fails |
| **Rate Limiter / Throttling**| Restricts number of requests to prevent overuse.                           | API gateway, backend protection                      |
| **Fail-Fast**                | Fails immediately on critical error, avoids cascading failures.            | Input validation, early resource checks              |
| **Fail-Safe**                | Allows graceful degradation without crashing.                              | Soft failure handling, e.g., ignore minor issues     |
| **Idempotency**              | Ensures repeated operations produce same results, avoiding duplication.    | Payment systems, retries                             |
| **Load Shedding**            | Drops requests when system is under pressure to preserve stability.        | Graceful handling during high load                   |
| **Health Checks / Heartbeats**| Monitors availability of services to detect failures early.               | Auto-healing, orchestration                         |
| **Caching**                  | Stores previous results to reduce dependency on live services.             | Database/service outages or performance tuning       |
| **Data Replication / Backup**| Keeps redundant data copies to ensure continuity after failure.            | Disaster recovery, high availability                 |
| **Chaos Engineering**        | Intentionally introduces failures to test system resiliency.               | Netflix’s Chaos Monkey, proactive fault tolerance    |

---

## **Popular Libraries That Implement These Patterns**

| Library / Framework      | Patterns Supported                                |
|--------------------------|---------------------------------------------------|
| **Resilience4j**         | Retry, Circuit Breaker, Rate Limiter, Bulkhead, Timeout, Fallback |
| **Spring Cloud Resilience4j** | Spring Boot integration for all above           |
| **Hystrix (deprecated)** | Circuit Breaker, Fallback, Thread Pool Isolation  |
| **Sentinel (by Alibaba)**| Flow control, Degrade, Circuit Breaker            |

---

## **Spring Boot Example (Circuit Breaker with Fallback)**

```java
@CircuitBreaker(name = "userService", fallbackMethod = "fallbackUser")
public User getUser(Long id) {
    return userClient.fetchUser(id); // Might fail
}

public User fallbackUser(Long id, Throwable ex) {
    return new User(id, "Default User");
}
```

---

Would you like a comparison table of these patterns or a demo Spring Boot project showing them in action?