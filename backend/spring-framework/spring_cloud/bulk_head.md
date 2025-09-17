Got it! Here’s a practical, Spring-Boot-friendly guide to the **Bulkhead Pattern** using **Resilience4j**—with explanations, Maven deps, config, and runnable code (sync + async/thread-pool variants).

---

# What is the Bulkhead Pattern?

Like watertight compartments on a ship, bulkheads **isolate resources** so a failure/surge in one area doesn’t sink the whole service. In microservices, that means **limiting concurrent calls** to a dependency or feature so it can’t exhaust threads/connections.

---

# When to use which bulkhead?

Resilience4j offers two types:

* **Semaphore Bulkhead** (in-JVM concurrency cap): limits *concurrent calls* to a section of code. Simple, low-overhead, for **CPU/light I/O** and **short latency** ops.
* **ThreadPool Bulkhead** (dedicated thread pool + queue): isolates calls on their own executor. Good for **potentially slow/variable I/O** to avoid blocking your main request threads.

---

# Maven dependencies (Spring Boot + Resilience4j)

```xml
<dependencies>
  <!-- Web (use WebFlux or MVC as you like) -->
  <dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-webflux</artifactId>
  </dependency>

  <!-- Resilience4j Spring Boot 2/3 integration -->
  <dependency>
    <groupId>io.github.resilience4j</groupId>
    <artifactId>resilience4j-spring-boot3</artifactId>
    <version>2.2.0</version>
  </dependency>

  <!-- (Optional) metrics + Actuator endpoints -->
  <dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-actuator</artifactId>
  </dependency>
  <dependency>
    <groupId>io.micrometer</groupId>
    <artifactId>micrometer-registry-prometheus</artifactId>
  </dependency>

  <!-- (Optional) validations/annotations -->
  <dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-validation</artifactId>
  </dependency>

  <!-- Test -->
  <dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-test</artifactId>
    <scope>test</scope>
  </dependency>
</dependencies>
```

---

# application.yml – define bulkheads

(blank line before table retained per your preference)

```yaml
server:
  port: 8080

management:
  endpoints:
    web:
      exposure:
        include: health,info,prometheus

resilience4j:
  bulkhead:
    instances:
      productApi:                # Semaphore bulkhead
        maxConcurrentCalls: 10   # reject the 11th concurrent call
        maxWaitDuration: 0       # 0 = fail fast; set e.g. 100ms to wait
      inventoryCalc:
        maxConcurrentCalls: 5
        maxWaitDuration: 50ms

  thread-pool-bulkhead:
    instances:
      slowPartnerApi:            # Dedicated pool + queue
        coreThreadPoolSize: 8
        maxThreadPoolSize: 16
        queueCapacity: 50
        keepAliveDuration: 30s
```

---

# Example: Service with **Semaphore Bulkhead**

```java
package com.example.bulkhead.service;

import io.github.resilience4j.bulkhead.annotation.Bulkhead;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Mono;

import java.time.Duration;

@Service
public class ProductService {

    // Simple reactive simulation of a remote call
    @Bulkhead(name = "productApi", type = Bulkhead.Type.SEMAPHORE, fallbackMethod = "getProductFallback")
    public Mono<String> getProduct(String id) {
        return Mono.defer(() -> Mono.just("PRODUCT-" + id))
                   .delayElement(Duration.ofMillis(80)); // pretend latency
    }

    // Fallback receives original args + exception
    private Mono<String> getProductFallback(String id, Throwable t) {
        return Mono.just("PRODUCT-FALLBACK-" + id);
    }
}
```

---

# Example: Service with **ThreadPool Bulkhead** (good for slow I/O)

```java
package com.example.bulkhead.service;

import io.github.resilience4j.bulkhead.annotation.Bulkhead;
import io.github.resilience4j.bulkhead.annotation.Bulkhead.Type;
import org.springframework.stereotype.Service;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

@Service
public class PartnerService {

    @Bulkhead(name = "slowPartnerApi", type = Type.THREADPOOL, fallbackMethod = "partnerFallback")
    public CompletableFuture<String> fetchPricing(String sku) {
        return CompletableFuture.supplyAsync(() -> {
            simulateSlowIo(200);  // simulate partner latency
            return "PRICE:" + sku + ":42.50";
        });
    }

    private void simulateSlowIo(long ms) {
        try { TimeUnit.MILLISECONDS.sleep(ms); } catch (InterruptedException ignored) {}
    }

    private CompletableFuture<String> partnerFallback(String sku, Throwable t) {
        return CompletableFuture.completedFuture("PRICE:" + sku + ":N/A");
    }
}
```

---

# Controller to exercise both

```java
package com.example.bulkhead.web;

import com.example.bulkhead.service.PartnerService;
import com.example.bulkhead.service.ProductService;
import org.springframework.web.bind.annotation.*;
import reactor.core.publisher.Mono;

import java.util.concurrent.CompletableFuture;

@RestController
@RequestMapping("/api")
public class DemoController {

    private final ProductService productService;
    private final PartnerService partnerService;

    public DemoController(ProductService productService, PartnerService partnerService) {
        this.productService = productService;
        this.partnerService = partnerService;
    }

    @GetMapping("/product/{id}")
    public Mono<String> getProduct(@PathVariable String id) {
        return productService.getProduct(id);
    }

    @GetMapping("/price/{sku}")
    public CompletableFuture<String> getPrice(@PathVariable String sku) {
        return partnerService.fetchPricing(sku);
    }
}
```

---

# Using WebClient with Bulkhead

If you call an external HTTP endpoint, you can wrap the call:

```java
package com.example.bulkhead.client;

import io.github.resilience4j.bulkhead.annotation.Bulkhead;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

@Component
public class PartnerClient {

    private final WebClient webClient = WebClient.builder().baseUrl("https://partner.example.com").build();

    @Bulkhead(name = "productApi", type = io.github.resilience4j.bulkhead.annotation.Bulkhead.Type.SEMAPHORE,
              fallbackMethod = "fallback")
    public Mono<String> fetchPartnerData(String id) {
        return webClient.get()
                .uri("/v1/items/{id}", id)
                .retrieve()
                .bodyToMono(String.class);
    }

    private Mono<String> fallback(String id, Throwable t) {
        return Mono.just("{\"id\":\"" + id + "\",\"source\":\"fallback\"}");
    }
}
```

---

# Programmatic (non-annotation) usage

Useful if you want dynamic names/partitions per tenant/endpoint.

```java
package com.example.bulkhead.core;

import io.github.resilience4j.bulkhead.Bulkhead;
import io.github.resilience4j.bulkhead.BulkheadConfig;
import io.vavr.control.Try;

import java.time.Duration;

public class CoreBulkheadExample {

    public String compute() {
        BulkheadConfig config = BulkheadConfig.custom()
                .maxConcurrentCalls(3)
                .maxWaitDuration(Duration.ofMillis(20))
                .build();

        Bulkhead bh = Bulkhead.of("compute-bh", config);

        return Try.ofSupplier(Bulkhead.decorateSupplier(bh, () -> "ok"))
                  .recover(throwable -> "fallback")
                  .get();
    }
}
```

---

# How Bulkhead differs from Circuit Breaker / Rate Limiter

(blank line before table)

| Pattern         | Primary Goal                         | Typical Trigger                       | Good For                                   |
| --------------- | ------------------------------------ | ------------------------------------- | ------------------------------------------ |
| Bulkhead        | **Isolate & cap concurrency**        | Too many parallel calls               | Protect thread pools; contain blast radius |
| Circuit Breaker | **Fail fast on persistent failures** | High error rate/latency over a window | Stop hammering a down dependency           |
| Rate Limiter    | **Cap request rate over time**       | Requests per second/minute thresholds | Fair usage & back-pressure                 |

You can use them together: e.g., **Bulkhead + CircuitBreaker + TimeLimiter** around a remote call.

---

# Observability & Tuning Tips

* Expose metrics: `/actuator/prometheus` → look for `resilience4j_bulkhead_*` and `resilience4j_thread_pool_bulkhead_*`.
* Start conservative: e.g., set `maxConcurrentCalls` to **(NCPU or connection slots)** and raise gradually.
* Prefer **ThreadPool Bulkhead** for slow/variable I/O; keep queues small to avoid long tail latency.
* Use **different bulkhead instances per dependency** (DB, cache, partner1, partner2).
* Apply **per-tenant/per-endpoint** names if you need stronger isolation.

---

# Quick load test idea (to see rejections)

Hammer `/api/product/{id}` with 50 parallel requests; with `maxConcurrentCalls: 10` you’ll see `BulkheadFullException` triggering the fallback for overflow calls.

---

If you want, I can tailor the config to your **specific downstreams** (e.g., DB, Kafka, partner APIs) or show a **JUnit test** that asserts rejections and fallback behavior.
