Let’s break down **IoC (Inversion of Control) container** in **Spring Boot**, its **features**, and how **ApplicationContext** and related concepts work during application startup — with code examples and practical explanation.

---

## **1. What is IoC (Inversion of Control) Container?**

The **IoC Container** in Spring is **responsible for creating objects**, **wiring dependencies**, **managing their lifecycle**, and **providing them when needed**.

### Key Interfaces:
- `BeanFactory` (basic container)
- `ApplicationContext` (enhanced container)

---

## **2. Features of the IoC Container**

| Feature                     | Description                                                      |
|----------------------------|------------------------------------------------------------------|
| **Dependency Injection**   | Injects dependencies into beans automatically                   |
| **Bean Lifecycle Handling**| Manages creation, initialization, and destruction of beans      |
| **Scope Management**       | Supports singleton, prototype, request, session, etc.           |
| **Event Publishing**       | Publishes events using `ApplicationEventPublisher`              |
| **Resource Loading**       | Loads resources like files, properties, etc.                    |
| **Internationalization**   | Provides i18n support via `MessageSource`                       |

---

## **3. Key Component: `ApplicationContext`**

It is the **central interface** that extends `BeanFactory` and adds more features:

```java
AnnotationConfigApplicationContext context =
    new AnnotationConfigApplicationContext(AppConfig.class);
```

---

## **4. IoC in Spring Boot App Startup**

When a Spring Boot app starts:

1. **`SpringApplication.run()`** is called
2. It **bootstraps** the **SpringApplicationContext**
3. All beans are **scanned, instantiated, and injected**
4. Lifecycle hooks (`@PostConstruct`, `InitializingBean`, etc.) are triggered
5. The app is now **ready to handle requests**

---

## **5. Code Example: IoC & ApplicationContext**

### a. Configuration & Bean Creation

```java
@Configuration
public class AppConfig {

    @Bean
    public EmployeeService employeeService() {
        return new EmployeeServiceImpl();
    }
}
```

### b. Service Interface and Implementation

```java
public interface EmployeeService {
    void printInfo();
}

public class EmployeeServiceImpl implements EmployeeService {
    public void printInfo() {
        System.out.println("EmployeeService: Hello from Spring IoC!");
    }
}
```

### c. Main App using IoC container

```java
public class MainApp {
    public static void main(String[] args) {
        ApplicationContext context =
            new AnnotationConfigApplicationContext(AppConfig.class);

        EmployeeService service = context.getBean(EmployeeService.class);
        service.printInfo();
    }
}
```

---

## **6. IoC and Spring Boot**

In Spring Boot, this is abstracted:

```java
@SpringBootApplication
public class MyApp {
    public static void main(String[] args) {
        SpringApplication.run(MyApp.class, args);
    }
}
```

### Internally:

- Scans components in the same or child packages.
- Creates an `ApplicationContext` (usually `AnnotationConfigServletWebServerApplicationContext`).
- Automatically wires beans using annotations like `@Autowired`, `@Component`, `@Service`, etc.

---

## **7. Summary of Concepts**

| Concept               | Role                                                                 |
|------------------------|----------------------------------------------------------------------|
| `@ComponentScan`      | Scans for Spring-managed beans                                        |
| `@Configuration`      | Defines configuration class for beans                                 |
| `@Bean`               | Declares a method as a bean definition                                |
| `ApplicationContext`  | Full-featured Spring container                                        |
| `BeanFactory`         | Basic IoC container                                                   |
| `@Autowired`          | Injects dependencies into fields, constructors, or setters            |
| `@PostConstruct`      | Lifecycle hook to run logic after bean creation                       |

---

Would you like to see how this behavior changes across different scopes (singleton vs prototype)? Or a breakdown of the bean lifecycle hooks (like `@PostConstruct`, `@PreDestroy`, `InitializingBean`)?