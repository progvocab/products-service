Here’s a precise, interview-ready answer:


---

Spring Boot Actuator

1. What it is:
Spring Boot Actuator is a set of production-ready features that allow you to monitor and manage your application. It provides endpoints for health checks, metrics, environment info, and more.


---

2. How it works:

Add the dependency:


<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-actuator</artifactId>
</dependency>

Spring Boot automatically registers built-in endpoints such as:

/actuator/health → checks application health

/actuator/metrics → exposes JVM, datasource, and custom metrics

/actuator/env → shows environment properties


Endpoints are enabled and configured via application.properties:


management.endpoints.web.exposure.include=health,metrics,env


---

3. Creating Custom Endpoints:
You can expose custom endpoints using @Endpoint or @RestController:

@Component
@Endpoint(id = "custom")
public class CustomEndpoint {

    @ReadOperation
    public String info() {
        return "Custom endpoint working!";
    }
}

Access via: /actuator/custom



---

4. Advantages:

Centralized monitoring and management

Supports prometheus, micrometer, and other monitoring tools

Can expose custom application-specific metrics and operations



---

In short:
Spring Boot Actuator provides built-in and customizable endpoints to monitor and manage applications in production, without adding much boilerplate code.


Here’s an example of typical responses for the three Actuator endpoints (/actuator/health, /actuator/metrics, /actuator/env) in a Spring Boot application:


---

1. /actuator/health

{
  "status": "UP",
  "components": {
    "db": {
      "status": "UP",
      "details": {
        "database": "PostgreSQL",
        "result": 1
      }
    },
    "diskSpace": {
      "status": "UP",
      "details": {
        "total": 499963174912,
        "free": 123456789012,
        "threshold": 10485760
      }
    }
  }
}

"status": "UP" indicates the application is healthy.

"components" shows individual health checks (DB, disk, etc.).



---

2. /actuator/metrics

{
  "names": [
    "jvm.memory.used",
    "jvm.memory.max",
    "process.cpu.usage",
    "system.cpu.usage",
    "tomcat.sessions.active.current"
  ],
  "measurements": [
    {
      "statistic": "VALUE",
      "value": 12345678
    }
  ]
}

"names" lists all available metrics.

"measurements" shows the value for a requested metric.

You can query a specific metric, e.g., /actuator/metrics/jvm.memory.used.



---

3. /actuator/env

{
  "activeProfiles": ["dev"],
  "propertySources": [
    {
      "name": "applicationConfig: [classpath:/application.yml]",
      "properties": {
        "server.port": {
          "value": "8080"
        },
        "spring.datasource.url": {
          "value": "jdbc:postgresql://localhost:5432/mydb"
        },
        "spring.datasource.username": {
          "value": "admin"
        }
      }
    },
    {
      "name": "systemEnvironment",
      "properties": {
        "JAVA_HOME": {
          "value": "/usr/lib/jvm/java-17"
        }
      }
    }
  ]
}

Shows active profiles.

Shows all property sources including environment variables and application configuration.



---

If you want, I can draw a simple diagram showing how these three endpoints relate to the Spring Boot application and its components. It’s very helpful for interviews.