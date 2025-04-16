In Spring, **you cannot have two beans with the same name**. Bean names must be **unique** in the application context. However, if you want to have **two beans of the same type**, you can distinguish them using `@Qualifier`.

Here’s how you can do it:

### 1. Define two beans of the same type with **different names**:

```java
@Configuration
public class AppConfig {

    @Bean(name = "firstService")
    public MyService myService1() {
        return new MyServiceImpl1();
    }

    @Bean(name = "secondService")
    public MyService myService2() {
        return new MyServiceImpl2();
    }
}
```

### 2. Inject using `@Qualifier`:

```java
@Component
public class MyComponent {

    private final MyService myService;

    public MyComponent(@Qualifier("secondService") MyService myService) {
        this.myService = myService;
    }

    public void doSomething() {
        myService.execute();
    }
}
```

### Summary:
- Bean **names must be unique**.
- You can have multiple beans of the same **type**, but with different **names** or **qualifiers**.
- Use `@Qualifier("beanName")` to disambiguate which bean to inject.

Let me know if you'd like to use `@Primary`, `@Profile`, or `@Qualifier` with annotations instead of in a config class.


Great question!

When using `@Service` (or `@Component`, `@Repository`, etc.), Spring will **automatically register the bean** with a default name — usually the **class name with the first letter in lowercase**. But you can **explicitly name the bean** by passing a value to the annotation, and then reference it using `@Qualifier`.

---

### **Example: Using `@Service` with Bean Name and `@Qualifier`**

#### Step 1: Define multiple implementations

```java
public interface MyService {
    void process();
}
```

```java
@Service("serviceA")
public class MyServiceImplA implements MyService {
    public void process() {
        System.out.println("Processing A");
    }
}

@Service("serviceB")
public class MyServiceImplB implements MyService {
    public void process() {
        System.out.println("Processing B");
    }
}
```

---

### Step 2: Inject with `@Qualifier`

```java
@Component
public class MyController {

    private final MyService myService;

    public MyController(@Qualifier("serviceB") MyService myService) {
        this.myService = myService;
    }

    public void execute() {
        myService.process();  // This will call MyServiceImplB
    }
}
```

---

### Alternate: Field Injection (not recommended, but possible)

```java
@Autowired
@Qualifier("serviceA")
private MyService myService;
```

---

### Recap

- You can give a **custom name** to a `@Service` by using: `@Service("customName")`
- Use `@Qualifier("customName")` to **inject** the specific implementation.
- Bean name in `@Qualifier` **must match** the name you set in `@Service`.

Let me know if you'd like an example using configuration class (`@Bean`) + `@Qualifier` as well.