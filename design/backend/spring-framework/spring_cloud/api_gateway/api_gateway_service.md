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

## 📂 Next Steps

Would you like a fallback controller, JWT auth filter, or integration with Spring Security? I can generate that next.
