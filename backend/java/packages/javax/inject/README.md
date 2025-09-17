The `javax.inject` package is part of the **Java Dependency Injection (DI) specification**, formally known as **JSR-330: Dependency Injection for Java**. It provides a **standard set of annotations** to support dependency injection across frameworks in a **framework-neutral** way.

---

### Purpose

The `javax.inject` package defines a minimal, general-purpose **dependency injection API** used by frameworks like:

* **Spring**
* **Google Guice**
* **Jakarta EE (formerly Java EE)**

It enables developers to write code that can work with any DI framework without being tightly coupled to one.

---

### Core Annotations in `javax.inject`

| Annotation     | Description                                                          |
| -------------- | -------------------------------------------------------------------- |
| `@Inject`      | Marks a constructor, field, or method for dependency injection.      |
| `@Named`       | Used to distinguish between multiple implementations (qualifiers).   |
| `@Qualifier`   | Meta-annotation to define custom qualifiers for injection.           |
| `@Singleton`   | Declares that a class is a singleton (only one instance will exist). |
| `@Scope`       | Meta-annotation to define custom scopes.                             |
| `@Provider<T>` | Interface to lazily or repeatedly retrieve instances of type `T`.    |

---

### Example Usage

```java
import javax.inject.Inject;
import javax.inject.Named;
import javax.inject.Singleton;

@Singleton
public class ServiceConsumer {

    private final PaymentService paymentService;

    @Inject
    public ServiceConsumer(@Named("paypal") PaymentService paymentService) {
        this.paymentService = paymentService;
    }

    public void process() {
        paymentService.pay();
    }
}
```

```java
public interface PaymentService {
    void pay();
}
```

```java
@Named("paypal")
public class PaypalPaymentService implements PaymentService {
    public void pay() {
        System.out.println("Paid using PayPal");
    }
}
```

---

### Differences from Spring's `@Autowired`

| Feature               | `@Inject` (`javax.inject`)           | `@Autowired` (Spring)   |
| --------------------- | ------------------------------------ | ----------------------- |
| Standard              | Yes (JSR-330)                        | No (Spring-specific)    |
| Qualifier             | Uses `@Named` or custom `@Qualifier` | Uses `@Qualifier`       |
| Optional dependencies | No native support                    | `required=false` option |
| Scopes                | Uses `@Scope` or `@Singleton`        | Spring-defined scopes   |

---

### Adoption

* Spring supports `javax.inject` annotations out of the box.
* Google Guice uses them natively.
* Useful in **framework-agnostic** libraries.

---

Let me know if you want to see how `javax.inject.Provider`, custom qualifiers, or scopes are used in detail.


Great! Let’s dive deeper into the advanced usage of the `javax.inject` package, covering:

1. **`@Inject` with `Provider<T>`**
2. **Custom `@Qualifier` annotations**
3. **Custom `@Scope` definitions**
4. **Comparison with Spring-style annotations**

---

### 1. `Provider<T>` – Lazy Injection or Multiple Instances

The `Provider<T>` interface allows for **on-demand instantiation** or **multiple injections**.

#### Example:

```java
import javax.inject.Inject;
import javax.inject.Provider;

public class EmailSender {

    @Inject
    private Provider<EmailService> emailServiceProvider;

    public void sendMultipleEmails() {
        EmailService service1 = emailServiceProvider.get();
        EmailService service2 = emailServiceProvider.get();
        // Two different instances if not singleton
        service1.send("First email");
        service2.send("Second email");
    }
}
```

---

### 2. Custom `@Qualifier` Annotation

When you have **multiple beans of the same type**, create a custom `@Qualifier` to distinguish between them.

#### Custom Qualifier:

```java
import javax.inject.Qualifier;
import java.lang.annotation.*;

@Qualifier
@Retention(RetentionPolicy.RUNTIME)
@Target({ FIELD, PARAMETER, METHOD })
public @interface Paypal {}
```

#### Usage:

```java
@Paypal
public class PaypalService implements PaymentService {
    public void pay() {
        System.out.println("Paying with PayPal");
    }
}
```

```java
public class PaymentProcessor {

    @Inject
    @Paypal
    private PaymentService service;

    public void process() {
        service.pay();
    }
}
```

---

### 3. Custom `@Scope` Annotation

A **scope** defines the lifecycle of injected objects. Custom scopes are usually supported in more full-featured DI frameworks like Guice or Spring.

#### Example of a Custom Scope:

```java
import javax.inject.Scope;
import java.lang.annotation.*;

@Scope
@Retention(RetentionPolicy.RUNTIME)
@Target({ TYPE, METHOD })
public @interface SessionScoped {}
```

> Note: Out of the box, `javax.inject` only defines `@Singleton`. For other scopes like request or session, your DI container (e.g., Guice, Spring, CDI) must implement the behavior.

---

### 4. Comparison with Spring Injection

| Feature               | `javax.inject` (JSR-330) | Spring Framework             |
| --------------------- | ------------------------ | ---------------------------- |
| Basic DI              | `@Inject`                | `@Autowired`                 |
| Qualifier             | `@Named`, `@Qualifier`   | `@Qualifier`                 |
| Optional dependencies | Not directly supported   | `@Autowired(required=false)` |
| Lazy injection        | `Provider<T>`            | `ObjectProvider<T>`          |
| Scopes                | `@Singleton`, custom     | `@Scope("...")`              |
| Framework Agnostic    | Yes                      | No                           |

---

Would you like to see a full working example using Guice or Spring with `javax.inject` annotations?
Here’s a **full working example using Google Guice** with `javax.inject` annotations.

---

## **Goal**: Inject different `PaymentService` implementations using a `@Qualifier`, managed by Guice.

### 1. **Define the Interfaces and Implementations**

```java
public interface PaymentService {
    void pay();
}
```

```java
public class CreditCardService implements PaymentService {
    public void pay() {
        System.out.println("Paid with Credit Card");
    }
}
```

```java
public class PaypalService implements PaymentService {
    public void pay() {
        System.out.println("Paid with PayPal");
    }
}
```

---

### 2. **Create a Custom Qualifier**

```java
import javax.inject.Qualifier;
import java.lang.annotation.*;

@Qualifier
@Retention(RetentionPolicy.RUNTIME)
@Target({ElementType.FIELD, ElementType.PARAMETER, ElementType.METHOD})
public @interface Paypal {}
```

```java
@Qualifier
@Retention(RetentionPolicy.RUNTIME)
@Target({ElementType.FIELD, ElementType.PARAMETER, ElementType.METHOD})
public @interface CreditCard {}
```

---

### 3. **Create a Guice Module**

```java
import com.google.inject.AbstractModule;
import com.google.inject.name.Names;

public class PaymentModule extends AbstractModule {
    @Override
    protected void configure() {
        bind(PaymentService.class).annotatedWith(Paypal.class).to(PaypalService.class);
        bind(PaymentService.class).annotatedWith(CreditCard.class).to(CreditCardService.class);
    }
}
```

---

### 4. **Create a Consumer Class**

```java
import javax.inject.Inject;

public class PaymentProcessor {

    private final PaymentService paypal;
    private final PaymentService creditCard;

    @Inject
    public PaymentProcessor(@Paypal PaymentService paypal,
                            @CreditCard PaymentService creditCard) {
        this.paypal = paypal;
        this.creditCard = creditCard;
    }

    public void processAllPayments() {
        paypal.pay();
        creditCard.pay();
    }
}
```

---

### 5. **Bootstrap with Guice Injector**

```java
import com.google.inject.Guice;
import com.google.inject.Injector;

public class Main {
    public static void main(String[] args) {
        Injector injector = Guice.createInjector(new PaymentModule());
        PaymentProcessor processor = injector.getInstance(PaymentProcessor.class);
        processor.processAllPayments();
    }
}
```

---

### **Output**

```
Paid with PayPal  
Paid with Credit Card
```

---

### Would You Like to See the Same Setup with Spring Framework?

