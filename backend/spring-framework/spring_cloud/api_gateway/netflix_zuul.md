Yes! Below is a **complete example** of using **Netflix Zuul** as an API Gateway to route requests to an **Employee microservice**.

> ⚠️ **Note**: Netflix Zuul 1 is **deprecated** in favor of **Spring Cloud Gateway**, but it's still used in legacy systems. Use it only if you're maintaining an older system.

---

## 🏗️ Goal

* Use **Zuul API Gateway** to forward `/employees/**` requests to `http://localhost:8081`
* Optionally add filters and logging
* Use Spring Boot + Spring Cloud Netflix Zuul

---

## ✅ Project Setup

### 📁 Project Structure

```
zuul-api-gateway/
├── pom.xml
└── src/
    └── main/
        ├── java/com/example/gateway/
        │   └── ZuulApiGatewayApplication.java
        └── resources/
            └── application.yml
```

---

## 📦 `pom.xml`

```xml
<project xmlns="http://maven.apache.org/POM/4.0.0" ...>
  <modelVersion>4.0.0</modelVersion>
  <groupId>com.example</groupId>
  <artifactId>zuul-api-gateway</artifactId>
  <version>1.0.0</version>
  <parent>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-parent</artifactId>
    <version>2.3.12.RELEASE</version> <!-- Compatible with Spring Cloud Netflix -->
  </parent>

  <dependencies>
    <!-- Zuul Starter -->
    <dependency>
      <groupId>org.springframework.cloud</groupId>
      <artifactId>spring-cloud-starter-netflix-zuul</artifactId>
    </dependency>

    <!-- Optional: Eureka for service discovery -->
    <!--
    <dependency>
      <groupId>org.springframework.cloud</groupId>
      <artifactId>spring-cloud-starter-netflix-eureka-client</artifactId>
    </dependency>
    -->

    <dependency>
      <groupId>org.springframework.boot</groupId>
      <artifactId>spring-boot-starter-web</artifactId>
    </dependency>
  </dependencies>

  <dependencyManagement>
    <dependencies>
      <dependency>
        <groupId>org.springframework.cloud</groupId>
        <artifactId>spring-cloud-dependencies</artifactId>
        <version>Hoxton.SR12</version>
        <type>pom</type>
        <scope>import</scope>
      </dependency>
    </dependencies>
  </dependencyManagement>
</project>
```

---

## 🚀 `ZuulApiGatewayApplication.java`

```java
package com.example.gateway;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cloud.netflix.zuul.EnableZuulProxy;

@SpringBootApplication
@EnableZuulProxy  // Enables Zuul reverse proxy
public class ZuulApiGatewayApplication {
    public static void main(String[] args) {
        SpringApplication.run(ZuulApiGatewayApplication.class, args);
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
    name: zuul-api-gateway

zuul:
  routes:
    employee-service:
      path: /employees/**
      url: http://localhost:8081  # target Employee microservice

  # Optional: prefix stripping
  strip-prefix: true

# Optional logging
logging:
  level:
    org.springframework.cloud.netflix.zuul: DEBUG
```

---

## 🧪 Example Requests

### 🎯 Hit via Gateway:

```http
GET http://localhost:8080/employees
```

Will route to:

```http
GET http://localhost:8081/employees
```

---

## ➕ Optional: Add Zuul Filter

### 📄 `PreFilter.java`

```java
@Component
public class PreFilter extends ZuulFilter {

    @Override
    public String filterType() {
        return "pre";
    }

    @Override
    public int filterOrder() {
        return 1; // Priority
    }

    @Override
    public boolean shouldFilter() {
        return true;
    }

    @Override
    public Object run() {
        HttpServletRequest request =
                RequestContext.getCurrentContext().getRequest();
        System.out.println("Incoming request: " + request.getRequestURI());
        return null;
    }
}
```

---

## ✅ Summary

| Feature                  | Supported in Zuul                       |
| ------------------------ | --------------------------------------- |
| Routing                  | ✅ Yes                                   |
| Path forwarding          | ✅ Yes                                   |
| Load Balancing           | ✅ With Eureka                           |
| Filters (pre/post/error) | ✅ Yes                                   |
| Reactive support         | ❌ No (Use Spring Cloud Gateway instead) |

---

## 🧠 Recommendation

* Use Zuul **only if maintaining legacy systems**
* For new projects, prefer **Spring Cloud Gateway** for modern reactive support, better routing, and resilience

---

Would you like the same gateway using **Eureka discovery** or **Spring Security JWT integration** with Zuul?
