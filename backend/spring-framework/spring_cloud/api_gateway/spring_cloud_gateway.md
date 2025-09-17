### ✅ API Gateway in Spring Cloud

In **Spring Cloud**, the API Gateway acts as a **single entry point** for routing requests to various microservices. It is typically implemented using **Spring Cloud Gateway**, which is a modern, reactive, and non-blocking API gateway built on **Spring WebFlux**.

---

## 🧱 What is an API Gateway?

An **API Gateway** is a **system design pattern** that sits between the client and microservices. It handles:

* **Routing**
* **Security**
* **Rate limiting**
* **Load balancing**
* **Monitoring**
* **Aggregation**

> It simplifies the client interaction with a microservice architecture by abstracting service details.

---

## 🚪 Spring Cloud Gateway Overview

**Spring Cloud Gateway** is the preferred API Gateway in the Spring ecosystem (replacing Netflix Zuul).

### 🔧 Basic Setup:

Add the dependency:

```xml
<dependency>
  <groupId>org.springframework.cloud</groupId>
  <artifactId>spring-cloud-starter-gateway</artifactId>
</dependency>
```

Basic config in `application.yml`:

```yaml
spring:
  cloud:
    gateway:
      routes:
        - id: user-service
          uri: http://localhost:8081
          predicates:
            - Path=/users/**
```

---

## ✨ Main Features of Spring Cloud Gateway

| Feature                          | Description                                                                       |
| -------------------------------- | --------------------------------------------------------------------------------- |
| **Routing**                      | Routes requests to microservices based on path, header, method, etc.              |
| **Predicate-based Filtering**    | Supports conditional routing using predicates like `Path`, `Host`, `Method`, etc. |
| **Filters (Pre/Post)**           | Modify request/response (e.g., add headers, strip path, transform body)           |
| **Load Balancing**               | Integrates with **Spring Cloud LoadBalancer** or **Eureka** for service discovery |
| **Path Rewriting**               | Easily rewrite incoming URLs                                                      |
| **Rate Limiting**                | Throttle clients using token bucket algorithm (via Redis)                         |
| **Circuit Breaker**              | Built-in support with **Resilience4j**                                            |
| **Authentication/Authorization** | Easily integrates with OAuth2 / JWT using Spring Security                         |
| **WebSocket support**            | Supports WebSocket proxying                                                       |
| **Reactive / Non-blocking**      | Built on Spring WebFlux for high concurrency                                      |

---

In **Spring Cloud Gateway**, **filters** are powerful components that allow you to **manipulate, control, and monitor** traffic going through the gateway. Filters can be **global** (applied to all routes) or **route-specific**, and they implement many key **system design patterns** like **rate limiting**, **circuit breaking**, **authentication**, and more.

---

## ✅ Categories of Filters in Spring Cloud Gateway

1. **Pre-filters** – run **before** the request is routed
2. **Post-filters** – run **after** the response is received

---

## 🚀 Common Route Filters (Declarative in `application.yml`)

| Filter Name                 | Purpose                                                |
| --------------------------- | ------------------------------------------------------ |
| `AddRequestHeader`          | Add a header before forwarding request                 |
| `AddResponseHeader`         | Add a header to the response                           |
| `StripPrefix`               | Remove path prefix before routing to downstream        |
| `RewritePath`               | Rewrite the path using regex                           |
| `RequestRateLimiter`        | Apply rate limiting (usually with Redis)               |
| `CircuitBreaker`            | Circuit breaker pattern (uses Resilience4j or Hystrix) |
| `Retry`                     | Retry failed requests automatically                    |
| `SetStatus`                 | Set the response status code directly                  |
| `RedirectTo`                | Redirect request to a new URL                          |
| `RequestHeaderToRequestUri` | Rewrite URI based on header                            |

---

## 🔒 1. `RequestRateLimiter` Filter

Implements **API rate limiting** using a token-bucket algorithm.

> Requires Redis.

### Example:

```yaml
filters:
  - name: RequestRateLimiter
    args:
      redis-rate-limiter.replenishRate: 5
      redis-rate-limiter.burstCapacity: 10
```

* `replenishRate`: tokens per second
* `burstCapacity`: max tokens in bucket

⛓️ Pattern: **Rate Limiting**

---

## 💥 2. `CircuitBreaker` Filter

Stops sending requests to a failing service temporarily to prevent cascading failures.

> Uses **Resilience4j** under the hood (default).

### Example:

```yaml
filters:
  - name: CircuitBreaker
    args:
      name: myCircuitBreaker
      fallbackUri: forward:/fallback
```

* `fallbackUri`: route used when circuit is open

⛓️ Pattern: **Circuit Breaker**, **Fail-Fast**

---

## 🔁 3. `Retry` Filter

Automatically retries failed requests.

### Example:

```yaml
filters:
  - name: Retry
    args:
      retries: 3
      statuses: BAD_GATEWAY
      methods: GET,POST
```

⛓️ Pattern: **Retry Pattern**

---

## ✂️ 4. `StripPrefix` Filter

Removes prefix from URL path before forwarding.

```yaml
filters:
  - StripPrefix=1
```

* `/api/employees` → `/employees`

⛓️ Pattern: **Path Rewriting**

---

## 🔃 5. `RewritePath` Filter

Uses regex to change the path.

```yaml
filters:
  - RewritePath=/api/(?<segment>.*), /$\{segment}
```

* `/api/employee` → `/employee`

⛓️ Pattern: **URL Rewrite**

---

## 🔧 6. `AddRequestHeader`, `AddResponseHeader`

Adds headers to request/response.

```yaml
filters:
  - AddRequestHeader=X-Gateway, GatewayService
  - AddResponseHeader=X-Processed-Time, ${date}
```

⛓️ Pattern: **Request Decoration**, **Observability**

---

## 🧪 7. `SetStatus`, `RedirectTo`, `RemoveRequestHeader`

Various control filters:

```yaml
- SetStatus=401
- RedirectTo=302, https://new-domain.com
- RemoveRequestHeader=Authorization
```

---

## 🧠 Global Filters

Custom or built-in filters applied to **all requests**, e.g.:

```java
@Bean
public GlobalFilter loggingFilter() {
    return (exchange, chain) -> {
        System.out.println("Incoming request: " + exchange.getRequest().getPath());
        return chain.filter(exchange);
    };
}
```

---

## 📦 Dependencies Required

To use `RequestRateLimiter` and `CircuitBreaker`, include:

```xml
<dependency>
  <groupId>org.springframework.cloud</groupId>
  <artifactId>spring-cloud-starter-gateway</artifactId>
</dependency>

<dependency>
  <groupId>org.springframework.boot</groupId>
  <artifactId>spring-boot-starter-data-redis-reactive</artifactId>
</dependency>

<dependency>
  <groupId>io.github.resilience4j</groupId>
  <artifactId>resilience4j-spring-boot2</artifactId>
</dependency>
```

---

## ✅ Summary Table

| Filter               | Purpose                     | Design Pattern         |
| -------------------- | --------------------------- | ---------------------- |
| `RequestRateLimiter` | Throttle requests           | Rate Limiting          |
| `CircuitBreaker`     | Fallback on service failure | Circuit Breaker        |
| `Retry`              | Auto retry failed calls     | Retry                  |
| `StripPrefix`        | Adjust request path         | URL Path Normalization |
| `RewritePath`        | Regex rewrite path          | URL Rewrite            |
| `AddRequestHeader`   | Add request metadata        | Request Decoration     |
| `AddResponseHeader`  | Enrich response headers     | Observability          |
| `SetStatus`          | Custom error code           | Response Control       |

---

Would you like a working Spring Boot API Gateway project with these filters configured for an employee microservice?


## 🧠 Major System Design Patterns it Implements

| System Design Pattern            | Role in Spring Cloud Gateway                                                |
| -------------------------------- | --------------------------------------------------------------------------- |
| **API Gateway Pattern**          | Central entry point for all client requests                                 |
| **Reverse Proxy**                | Forwards requests to appropriate services                                   |
| **Facade Pattern**               | Hides complexity of backend service interactions                            |
| **Gateway Aggregation**          | Aggregates responses from multiple services (manually via filters/handlers) |
| **Service Discovery**            | Discovers service instances (via Eureka, Consul)                            |
| **Circuit Breaker / Resilience** | Handles failure gracefully with retries/fallbacks                           |
| **Security Gateway**             | Centralized authentication/authorization via filters                        |

---

Great! Let's dive into some of the **most common and powerful tools** in **Spring Cloud Gateway**: `Predicate`, `StripPrefix`, `AddRequestHeader`, and `RewritePath`.

These are used to **match**, **transform**, and **enrich** requests as they pass through your API Gateway before reaching downstream services.

---

## ✅ 1. `Predicate` – *When should this route apply?*

**Predicates** define **conditions** that must be **true** for a route to be matched.

> Think of predicates as **if statements** for routing.

### 🔸 Example: Path Predicate

```yaml
predicates:
  - Path=/employees/**
```

🔍 This means:

* The route will only be selected if the incoming request URI starts with `/employees/`.

Other common predicates:

| Predicate | Example             | Matches If...                       |
| --------- | ------------------- | ----------------------------------- |
| `Path`    | `/users/**`         | Path starts with `/users/`          |
| `Method`  | `GET`               | HTTP method is GET                  |
| `Header`  | `X-Request-ID, \d+` | Header exists and matches pattern   |
| `Host`    | `**.example.com`    | Host matches the wildcard domain    |
| `Query`   | `debug=true`        | Query param `debug=true` is present |

---

## ✅ 2. `StripPrefix` Filter – *Remove parts of the path*

```yaml
filters:
  - StripPrefix=1
```

🔍 This means:

* If the request is `/employees/123`
* The gateway will **strip 1 path segment** → forward `/123` to the backend

💡 Useful when your downstream service does **not expect the prefix** used for routing.

| Incoming Path    | StripPrefix=1 | Forwarded Path |
| ---------------- | ------------- | -------------- |
| `/employees/1`   | Yes           | `/1`           |
| `/api/users/abc` | StripPrefix=2 | `/abc`         |

---

## ✅ 3. `AddRequestHeader` Filter – *Inject new headers into the request*

```yaml
filters:
  - AddRequestHeader=X-Gateway-Source, spring-cloud-gateway
```

🔍 This means:

* Adds header: `X-Gateway-Source: spring-cloud-gateway` to the request forwarded to backend

This is often used to:

* Add tracking headers
* Pass tenant ID, roles, or user info
* Mark request as coming via gateway

---

## ✅ 4. `RewritePath` Filter – *Change the request URI using regex*

```yaml
filters:
  - RewritePath=/employees/(?<id>.*), /api/v1/employees/${id}
```

🔍 This means:

* If request is `/employees/123`
* The gateway rewrites it to `/api/v1/employees/123` before forwarding

### 💡 Use Case:

Your external API is `/employees/123`, but internal microservice expects `/api/v1/employees/123`.

| Incoming Path  | Rewrite Rule                                       | Forwarded Path        |
| -------------- | -------------------------------------------------- | --------------------- |
| `/employees/5` | `/employees/(?<id>.*)` → `/api/v1/employees/${id}` | `/api/v1/employees/5` |

---

## 🎯 Combining Everything – Example Route

```yaml
spring:
  cloud:
    gateway:
      routes:
        - id: employee-service
          uri: http://localhost:8081
          predicates:
            - Path=/employees/**
          filters:
            - StripPrefix=1
            - AddRequestHeader=X-Source, gateway
            - RewritePath=/employees/(?<id>.*), /internal/employee/${id}
```

### What this route does:

1. Matches paths like `/employees/**`
2. Strips `/employees` from the request path
3. Adds a custom header `X-Source: gateway`
4. Rewrites path to match internal structure like `/internal/employee/{id}`

---

## ✅ Summary Table

| Name               | Type     | Purpose                                |
| ------------------ | -------- | -------------------------------------- |
| `Predicate`        | Matching | Decides **when** to apply the route    |
| `StripPrefix`      | Filter   | Removes N segments from the path       |
| `AddRequestHeader` | Filter   | Adds a header to the forwarded request |
| `RewritePath`      | Filter   | Modifies request URI using regex       |

---


## 📘 Example: Routing with Filters and Rate Limiting

```yaml
spring:
  cloud:
    gateway:
      routes:
        - id: order-service
          uri: lb://ORDER-SERVICE
          predicates:
            - Path=/orders/**
          filters:
            - name: RequestRateLimiter
              args:
                redis-rate-limiter.replenishRate: 10
                redis-rate-limiter.burstCapacity: 20
            - name: CircuitBreaker
              args:
                name: orderCB
                fallbackUri: forward:/fallback/order
```

---

## ✅ Benefits in Microservices Architecture

| Benefit                             | Description                                  |
| ----------------------------------- | -------------------------------------------- |
| **Simplifies client logic**         | Client talks to one endpoint instead of many |
| **Centralized control**             | Auth, logging, and routing in one place      |
| **Decouples clients from services** | Services can evolve independently            |
| **Enables cross-cutting concerns**  | Like auth, rate limiting, tracing            |

---

## 🔚 Summary

| Feature                   | Spring Cloud Gateway    |
| ------------------------- | ----------------------- |
| Routing, Filters          | ✅ Yes                   |
| Load balancing, Discovery | ✅ Yes                   |
| JWT/OAuth2 security       | ✅ Yes                   |
| Circuit breaking          | ✅ Yes                   |
| Rate limiting             | ✅ Yes                   |
| Aggregation               | 🟡 Manual only          |
| Reactive support          | ✅ Yes (WebFlux)         |
| Zuul alternative          | ✅ Replaces Netflix Zuul |

---

Here’s a complete Spring Boot **API Gateway project** using **Spring Cloud Gateway**, designed to **route traffic to an existing `employees` microservice**.

---

## ✅ Requirements

* `employees` microservice is running at: `http://localhost:8081`
* Gateway should:

  * Route all `/employees/**` traffic to it
  * Add logging and optional rate limiting
  * Support service discovery (optional)

---

## 📁 Project Structure

```
api-gateway/
├── pom.xml
└── src/
    └── main/
        ├── java/com/example/gateway/
        │   └── ApiGatewayApplication.java
        └── resources/
            └── application.yml
```

---

## 📦 `pom.xml`

```xml
<project xmlns="http://maven.apache.org/POM/4.0.0" ...>
  <modelVersion>4.0.0</modelVersion>
  <groupId>com.example</groupId>
  <artifactId>api-gateway</artifactId>
  <version>1.0.0</version>
  <parent>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-parent</artifactId>
    <version>3.2.4</version>
  </parent>

  <dependencies>
    <!-- Spring Cloud Gateway -->
    <dependency>
      <groupId>org.springframework.cloud</groupId>
      <artifactId>spring-cloud-starter-gateway</artifactId>
    </dependency>

    <!-- For Logging -->
    <dependency>
      <groupId>org.springframework.boot</groupId>
      <artifactId>spring-boot-starter-actuator</artifactId>
    </dependency>

    <!-- Optional: For service discovery via Eureka -->
    <!-- <dependency>
      <groupId>org.springframework.cloud</groupId>
      <artifactId>spring-cloud-starter-netflix-eureka-client</artifactId>
    </dependency> -->

    <!-- Optional: Rate Limiting via Redis -->
    <!--
    <dependency>
      <groupId>org.springframework.boot</groupId>
      <artifactId>spring-boot-starter-data-redis-reactive</artifactId>
    </dependency>
    -->

  </dependencies>

  <dependencyManagement>
    <dependencies>
      <dependency>
        <groupId>org.springframework.cloud</groupId>
        <artifactId>spring-cloud-dependencies</artifactId>
        <version>2023.0.1</version>
        <type>pom</type>
        <scope>import</scope>
      </dependency>
    </dependencies>
  </dependencyManagement>
</project>
```

---

## 🚀 `ApiGatewayApplication.java`

```java
package com.example.gateway;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class ApiGatewayApplication {
    public static void main(String[] args) {
        SpringApplication.run(ApiGatewayApplication.class, args);
    }
}
```

---

## ⚙️ `application.yml`

```yaml
server:
  port: 8080

spring:
  application:
    name: api-gateway
  cloud:
    gateway:
      routes:
        - id: employee-service
          uri: http://localhost:8081
          predicates:
            - Path=/employees/**
          filters:
            - AddRequestHeader=X-Gateway-Tag, Employee-Gateway
            - RewritePath=/employees/(?<segment>.*), /${segment}

# Optional for logging
logging:
  level:
    org.springframework.cloud.gateway: DEBUG
```

---

## 🧪 Test It

Assume your `employees` service exposes:

```
GET http://localhost:8081/employees
```

Then hit the API Gateway:

```
GET http://localhost:8080/employees
```

You should receive the same response.

---

## 🧠 Optional Enhancements

### ✅ Enable Service Discovery (if using Eureka):

* Uncomment Eureka dependency in `pom.xml`
* Change `uri: lb://EMPLOYEE-SERVICE` in `application.yml`
* Register `employees` service with Eureka

### ✅ Add Rate Limiting:

Add to filters:

```yaml
- name: RequestRateLimiter
  args:
    redis-rate-limiter.replenishRate: 5
    redis-rate-limiter.burstCapacity: 10
```

### ✅ Add Circuit Breaker:

```yaml
- name: CircuitBreaker
  args:
    name: employeeCircuitBreaker
    fallbackUri: forward:/fallback/employees
```

---


