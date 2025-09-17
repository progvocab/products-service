### ✅ What is a **Eureka Server**?

**Eureka Server** is a **Service Registry** from **Netflix OSS**, part of the **Spring Cloud Netflix** stack.

In a microservices architecture, **services need to discover each other** (for example, Employee service talking to Department service). Instead of hardcoding IP addresses or URLs, services register themselves with Eureka and discover each other using service names.

---

### 🧠 Analogy

* **Eureka Server** = Phonebook
* **Employee Service** = Registers its number (URL) with the Phonebook
* **Department Service** = Looks up Employee service from the Phonebook to call it

---

## ✅ Fully Working Code for Eureka Server

We’ll use:

* Spring Boot 2.7.x
* Spring Cloud 2021.x

---

### 📁 Project Structure

```
eureka-server/
├── src/
│   └── main/
│       ├── java/com/example/eurekaserver/
│       │   └── EurekaServerApplication.java
│       └── resources/
│           └── application.properties
└── pom.xml
```

---

### 1️⃣ `pom.xml`

```xml
<project xmlns="http://maven.apache.org/POM/4.0.0" ...>
  <modelVersion>4.0.0</modelVersion>
  <groupId>com.example</groupId>
  <artifactId>eureka-server</artifactId>
  <version>1.0.0</version>
  <packaging>jar</packaging>

  <parent>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-parent</artifactId>
    <version>2.7.18</version>
  </parent>

  <dependencies>
    <dependency>
      <groupId>org.springframework.cloud</groupId>
      <artifactId>spring-cloud-starter-netflix-eureka-server</artifactId>
    </dependency>
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
        <version>2021.0.9</version>
        <type>pom</type>
        <scope>import</scope>
      </dependency>
    </dependencies>
  </dependencyManagement>

</project>
```

---

### 2️⃣ `EurekaServerApplication.java`

```java
package com.example.eurekaserver;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cloud.netflix.eureka.server.EnableEurekaServer;

@SpringBootApplication
@EnableEurekaServer
public class EurekaServerApplication {
    public static void main(String[] args) {
        SpringApplication.run(EurekaServerApplication.class, args);
    }
}
```

---

### 3️⃣ `application.properties`

```properties
server.port=8761
spring.application.name=eureka-server

eureka.client.register-with-eureka=false
eureka.client.fetch-registry=false
```

---

### ▶️ How to Run

1. Run `EurekaServerApplication.java`
2. Open browser: [http://localhost:8761](http://localhost:8761)
3. You’ll see a Eureka dashboard.

---

### 🧪 Test with a Client

To test, add Eureka Client dependency to another Spring Boot app like `employee-service`:

```properties
spring.application.name=employee-service
server.port=8081
eureka.client.service-url.defaultZone=http://localhost:8761/eureka
```

Add this dependency to its `pom.xml`:

```xml
<dependency>
  <groupId>org.springframework.cloud</groupId>
  <artifactId>spring-cloud-starter-netflix-eureka-client</artifactId>
</dependency>
```

And annotate with `@EnableEurekaClient`.

---

Would you like a GitHub repo or a ZIP of this Eureka Server?
