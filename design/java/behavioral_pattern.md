**Behavioral design patterns** focus on how **objects interact and communicate**, and how responsibility is **delegated between them**. Java and Spring Boot make heavy use of these patterns, either directly in the JDK or within the framework’s core architecture.

---

### **1. Strategy Pattern**

**Purpose:** Define a family of algorithms, encapsulate each one, and make them interchangeable.

**In Java:**
- `Comparator` interface
- `Runnable`, `Callable`

**In Spring Boot:**
- `@Conditional` beans
- Custom implementations of interfaces like `AuthenticationProvider`, `Converter`, etc.

---

### **2. Observer Pattern**

**Purpose:** Notify multiple objects when the state of one object changes.

**In Java:**
- `java.util.Observer` (deprecated now)
- Event listeners (e.g., `ActionListener`)

**In Spring Boot:**
- `ApplicationEventPublisher`
- `@EventListener` methods
- Spring’s event-driven architecture

---

### **3. Template Method Pattern**

**Purpose:** Define the skeleton of an algorithm in a method, deferring some steps to subclasses.

**In Java:**
- `AbstractList`, `AbstractMap`
- JDBC template-style code

**In Spring Boot:**
- `JdbcTemplate`, `RestTemplate`, `WebSecurityConfigurerAdapter` (pre-Spring Security 6)
- You extend and override hooks (`configure()`, etc.)

---

### **4. Chain of Responsibility**

**Purpose:** Pass a request along a chain of handlers.

**In Java:**
- Servlet Filters
- Logging frameworks

**In Spring Boot:**
- Spring Security filters (`FilterChainProxy`)
- Spring WebFlux filter chains
- Exception resolvers (`HandlerExceptionResolver`)

---

### **5. Command Pattern**

**Purpose:** Encapsulate a request as an object.

**In Java:**
- `Runnable` and `Callable` as commands
- `TimerTask`

**In Spring Boot:**
- Task execution with `@Async`
- Scheduled tasks with `@Scheduled`

---

### **6. Mediator Pattern**

**Purpose:** Reduce complexity by centralizing communication between objects.

**In Spring Boot:**
- Spring’s **ApplicationContext** plays this role
- Messaging systems (e.g., Spring Integration, Spring Cloud Stream)
- Event publishing decouples sender and receiver

---

### **7. State Pattern**

**Purpose:** Change behavior of an object when its internal state changes.

**In Java:**
- State machines
- Enum-based switch logic

**In Spring Boot:**
- Spring State Machine project
- State-specific services or components

---

### **8. Iterator Pattern**

**Purpose:** Provide a way to access elements without exposing the underlying structure.

**In Java:**
- `Iterator`, `Iterable`

**In Spring Boot:**
- Streams, pagination (`Page<T>`)
- Reactive streams in WebFlux (`Flux`, `Mono`)

---

### **9. Visitor Pattern**

**Purpose:** Add new operations to objects without modifying their structure.

**Less common**, but found in:
- Reflection-based logic (Jackson, Hibernate)
- BeanPostProcessors or aspect-based visitation

---

### Summary Table

**Blank line before table for markdown:**

| Pattern               | Java Usage                           | Spring Boot Usage                         |
|------------------------|---------------------------------------|--------------------------------------------|
| Strategy              | `Comparator`, `Runnable`              | Bean selection, service impls              |
| Observer              | Event listeners                       | `ApplicationEvent`, `@EventListener`       |
| Template Method       | `Abstract*` classes                   | `RestTemplate`, `JdbcTemplate`             |
| Chain of Responsibility | Servlet filters                     | Security filter chain, exception handling  |
| Command               | `Runnable`, `Callable`                | `@Async`, `@Scheduled`                     |
| Mediator              | -                                     | `ApplicationContext`, messaging            |
| State                 | Enums, custom logic                   | Spring State Machine                       |
| Iterator              | `Iterator`, `Stream`                  | Pagination, Reactive Flux                  |
| Visitor               | Reflection, visitors                  | Bean processing, AOP                       |

---

Want a visual representation or a real-world example from Spring (like event handling or strategy-based bean selection)?