Let's dive into it properly:

---

# **Bean Scopes in Spring Boot**

When you create a **bean** (`@Component`, `@Service`, `@Bean`, etc.) in Spring, it **controls the lifecycle** of that bean.  
**Scope** decides **how many instances** of a bean **exist** and **how they are shared**.

---

# **Supported Bean Scopes in Spring Boot**

| Scope | Description | Example |
|:------|:------------|:--------|
| `singleton` (default) | One instance per Spring container | Service classes, Repositories |
| `prototype` | New instance every time it is requested | Non-shared stateful beans |
| `request` | One instance per HTTP request (only in Web apps) | Controller-related data |
| `session` | One instance per HTTP session | User session data |
| `application` | One instance per ServletContext | Application-level cache |
| `websocket` | One instance per WebSocket session | WebSocket communication |

---

# **Code Examples for Each Scope**

### 1. **Singleton Scope (Default)**

```java
@Component
public class SingletonService {
    public SingletonService() {
        System.out.println("SingletonService created");
    }
}
```

Use case: Shared, **stateless** services (like EmployeeService).

**No special config needed** — it's default.

---

### 2. **Prototype Scope**

```java
@Component
@Scope("prototype")
public class PrototypeService {
    public PrototypeService() {
        System.out.println("PrototypeService created");
    }
}
```

**Use case:**  
- Heavy objects  
- Objects with temporary, **stateful** data (e.g., file uploads, game moves)

When you inject `PrototypeService`, **new object created every time**.

---

### 3. **Request Scope (Web Apps)**

```java
@Component
@Scope("request")
public class RequestService {
    public RequestService() {
        System.out.println("RequestService created");
    }
}
```

**Use case:**  
- Store **per-request data**.
- Useful in web controllers handling user input.

---

### 4. **Session Scope**

```java
@Component
@Scope("session")
public class SessionService {
    public SessionService() {
        System.out.println("SessionService created");
    }
}
```

**Use case:**  
- **User login sessions**.
- Shopping cart during browsing.

---

### 5. **Application Scope**

```java
@Component
@Scope("application")
public class ApplicationService {
    public ApplicationService() {
        System.out.println("ApplicationService created");
    }
}
```

**Use case:**  
- Shared cache/data at application-wide level.

---

### 6. **Websocket Scope**

```java
@Component
@Scope("websocket")
public class WebSocketService {
    public WebSocketService() {
        System.out.println("WebSocketService created");
    }
}
```

**Use case:**  
- Store per-WebSocket connection data in chat apps, gaming apps.

---

# **How to inject Prototype scoped Bean into Singleton safely?**

Because **singleton** beans are created once but **prototype** beans should be **fresh each time**,  
you can’t inject `@Autowired` directly.  
Use **ObjectProvider** or **Provider<T>**.

```java
@Component
public class SingletonUsingPrototype {

    @Autowired
    private ObjectProvider<PrototypeService> prototypeProvider;

    public void doSomething() {
        PrototypeService service = prototypeProvider.getObject();
        service.toString();
    }
}
```

---

# **Design Patterns Related to Bean Scopes**

| Design Pattern | How it Relates |
|:---------------|:--------------|
| Singleton | Default Spring beans are Singletons |
| Factory Method | Spring internally uses factories to create and manage beans |
| Prototype | Prototype scope follows Prototype design pattern (new instance per request) |
| Dependency Injection | All beans managed through DI |
| Service Locator (optional) | Spring context can be seen as a kind of service locator |

---

# **Use Cases Based on Scopes**

| Scope | Typical Use Case |
|:------|:-----------------|
| Singleton | Service classes, Database connections |
| Prototype | Complex data models, temporary calculation |
| Request | Per HTTP request, API token validation |
| Session | User shopping cart, preferences |
| Application | Global configuration store |
| Websocket | Multiplayer game rooms, chat sessions |

---

# **Summary**

| Feature | Details |
|:--------|:--------|
| Default Scope | Singleton |
| Annotation to change | `@Scope("scopeName")` |
| Important Patterns | Singleton, Factory, Prototype, Dependency Injection |
| Special Handling | Injecting prototype into singleton needs `ObjectProvider` |

---

# Quick Tip:

If you want to **programmatically define scope**:

```java
@Bean
@Scope(ConfigurableBeanFactory.SCOPE_PROTOTYPE)
public MyBean myBean() {
    return new MyBean();
}
```

---

Would you also like me to show you:  
> **"Advanced scenarios like refreshing a request-scoped bean inside a singleton controller?"**  
This happens a lot in complex Spring Boot APIs.  
(**Yes/No?**)  
