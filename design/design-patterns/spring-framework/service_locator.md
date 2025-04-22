The **Service Locator** pattern is a **design pattern** used to **decouple service consumers from service implementations** by centralizing the logic to locate and provide service instances. Spring Framework provides this capability, but it encourages **Dependency Injection (DI)** as the preferred alternative.

---

## **1. What is the Service Locator Pattern?**

- A class or interface acts as a registry where services are **registered** and **retrieved** by name or type.
- It hides the logic of creating or locating dependencies.

### **When to use it?**
- When dynamic service resolution is required at runtime.
- When services are determined based on **external input** or **configuration**.

---

## **2. Service Locator in Spring Framework**

Spring supports this via:
- `ServiceLocatorFactoryBean`
- `ApplicationContext` (not recommended directly for injection)

### **Example using `ServiceLocatorFactoryBean`:**

#### Step 1: Define the Service Interface

```java
public interface PaymentService {
    void processPayment();
}
```

#### Step 2: Implementations

```java
@Component("creditCardService")
public class CreditCardPaymentService implements PaymentService {
    public void processPayment() {
        System.out.println("Processing credit card payment.");
    }
}

@Component("paypalService")
public class PaypalPaymentService implements PaymentService {
    public void processPayment() {
        System.out.println("Processing PayPal payment.");
    }
}
```

#### Step 3: Define a Service Locator Interface

```java
public interface PaymentServiceFactory {
    PaymentService getService(String serviceName);
}
```

#### Step 4: Register `ServiceLocatorFactoryBean` in a Config Class

```java
@Configuration
public class ServiceLocatorConfig {

    @Bean
    public ServiceLocatorFactoryBean paymentServiceFactory() {
        ServiceLocatorFactoryBean factoryBean = new ServiceLocatorFactoryBean();
        factoryBean.setServiceLocatorInterface(PaymentServiceFactory.class);
        return factoryBean;
    }
}
```

#### Step 5: Use It

```java
@Autowired
private PaymentServiceFactory paymentServiceFactory;

public void pay(String type) {
    PaymentService service = paymentServiceFactory.getService(type);
    service.processPayment();
}
```

---

## **3. Using `ApplicationContext` as Service Locator (Discouraged)**

```java
@Autowired
private ApplicationContext context;

public void dynamicServiceCall(String beanName) {
    PaymentService service = (PaymentService) context.getBean(beanName);
    service.processPayment();
}
```

> **Note**: This creates a tight coupling to the framework and makes testing harder. Use **DI** or `ServiceLocatorFactoryBean` instead.

---

## **4. Service Locator in Spring Boot**

Spring Boot uses the **same principles** as Spring Framework:

- It discourages the Service Locator pattern in favor of **constructor-based dependency injection**.
- But when needed, `ApplicationContext` or `ServiceLocatorFactoryBean` can be used similarly.

### **Best Practice in Spring Boot:**

Use constructor injection:

```java
@Service
public class PaymentHandler {
    private final CreditCardPaymentService creditCardService;

    public PaymentHandler(CreditCardPaymentService creditCardService) {
        this.creditCardService = creditCardService;
    }

    public void handle() {
        creditCardService.processPayment();
    }
}
```

---

## **5. Summary:**

| Aspect                  | Service Locator                    | Dependency Injection (Preferred)   |
|-------------------------|------------------------------------|------------------------------------|
| Flexibility             | High (choose at runtime)           | Fixed at injection time            |
| Coupling                | High (tight to Spring)             | Low                                |
| Testability             | Harder                             | Easier                             |
| Spring Support          | Via `ServiceLocatorFactoryBean`    | Native                             |

---

Would you like a runnable Spring Boot example that demonstrates both approaches: Service Locator vs Dependency Injection?