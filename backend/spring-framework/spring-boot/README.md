# Spring Boot

 
Running a Spring Boot JAR triggers:

1. JVM starts
2. Spring Boot custom classloader loads nested JARs
3. `@SpringBootApplication` main class runs
4. Spring initializes environment
5. Component scanning
6. Auto-configuration
7. Bean creation + dependency injection
8. Embedded server startup
9. DispatcherServlet initialization
10. ApplicationReadyEvent
11. App runs and serves requests

```
java -jar app.jar
   |
   v
JVM starts
   |
   v
Spring Boot Launcher loads nested JARs
   |
   v
Your main class runs
   |
   v
SpringApplication.run()
   |
   v
Create ApplicationContext
   |
   v
Load environment + configs
   |
   v
Component scanning
   |
   v
Auto-configuration (classpath-based)
   |
   v
Instantiate and wire beans
   |
   v
Start embedded server (Tomcat/Netty)
   |
   v
Initialize DispatcherServlet
   |
   v
Fire lifecycle events
   |
   v
APPLICATION READY 
```



```
java -jar myapp-1.0.jar
```

###   **1. JVM Starts**

* The JVM starts a new process.
* It loads the JAR file.
* Finds the `Main-Class` in `META-INF/MANIFEST.MF`.

Example:

```
Main-Class: org.springframework.boot.loader.JarLauncher
```

For fat JARs, Spring Boot uses its custom launcher.


###  **2. Spring Boot Launcher Kicks In**

Spring Boot has a custom **classloader**:
`org.springframework.boot.loader.LaunchedURLClassLoader`

It:

* Loads your app classes
* Loads dependency JARs nested inside `BOOT-INF/lib/`
* Loads your application main class from `BOOT-INF/classes/`

This works because Boot JARs are **nested JARs**, not standard.


### **3. Main Application Class Runs**

Spring Boot finds your `@SpringBootApplication` class:

```java
@SpringBootApplication
public class MyApp {
    public static void main(String[] args) {
        SpringApplication.run(MyApp.class, args);
    }
}
```

This triggers Spring Boot's bootstrap process.


➤ `ApplicationStartingEvent` triggered 
- Fired immediately when the SpringApplication starts, before environment or context are created.

➤ `ApplicationEnvironmentPreparedEvent` triggered
- Environment is prepared (properties, profiles loaded), but ApplicationContext is not created yet.

### **4. SpringApplication Bootstraps the Application**

`SpringApplication.run()` performs many tasks:

### 4.1 Create ApplicationContext

Depending on the app type:

* Web app → `AnnotationConfigServletWebServerApplicationContext`
* Reactive app → `ReactiveWebServerApplicationContext`
* Non-web app → `AnnotationConfigApplicationContext`
  
➤ `ApplicationContextInitializedEvent` triggered
- ApplicationContext object is created ✓ , but no beans have been loaded yet. ⟳

### 4.2 Start Spring Environment

Loads:

* `application.properties` or `.yaml`
* OS env variables
* JVM parameters
* Profiles (`spring.profiles.active`)
* External config files

### **5. Component Scanning**

Spring scans your package:

```
com.example.myapp.*
```

Finds:

* `@Component`
* `@Service`
* `@Repository`
* `@Controller`
* `@RestController`
* `@Configuration`
* `@Bean` methods

Registers them into the **IoC container**.

➤ `ApplicationPreparedEvent` triggered 
- Bean definitions are loaded ✓ BUT no beans are created yet. ⟳

### **6. Auto-Configuration Starts (`@EnableAutoConfiguration`)**

Spring Boot now checks the classpath and applies auto-configurations.

Example:

* `spring-boot-starter-web` → setup Tomcat + MVC
* `spring-boot-starter-data-jpa` → Hibernate + DataSource
* `spring-boot-starter-security` → Security auto-setup
* `spring-boot-starter-actuator` → Monitoring endpoints

These are triggered by `spring.factories` (Spring Boot 2) or `META-INF/spring/org.springframework.boot.autoconfigure.AutoConfiguration.imports` (Boot 3).


### **7. Beans Are Instantiated (Dependency Injection)**

Spring now creates all beans:

* Constructor injection
* Setter injection
* Field injection (not recommended)

[Bean lifecycle](../core/bean_lifecycle.md)

1. Instantiate object - Constructor called
2. Populate dependencies - @Autowired
3. BeanPostProcessors .postProcessBeforeInitialization
   - modifying bean properties
   - injecting additional metadata
   - wrapping with helper decorators
5. Call `@PostConstruct` methods 
6. Initialize bean : InitializingBean.afterPropertiesSet()
7. @Bean(initMethod="...")
8.  BeanPostProcessor.postProcessAfterInitialization() : This is usually where actual bean replacement happens (proxying).
    - Spring AOP creates proxies
    - @Transactional proxies are generated
    - @Async proxy wrappers added
    - Security proxy layers added
      
➤ `ContextRefreshedEvent` triggered 
- Bean factory setup ✓
- BeanPostProcessors registered ✓
- All singleton beans created ✓
- Web server not started yet ⟳

  
### **8. Web Server Starts (for web apps)**

If your app is a web application, Spring Boot starts embedded servers

* `MVC` : Tomcat (default) , Jetty , Undertow
* `WebFlux` : Netty  

Server steps:

* Create server instance
* Bind port (default: **8080**)
* Initialize servlet context
* Register DispatcherServlet

➤ `ServletWebServerInitializedEvent` (Spring MVC)
<br/> OR <br/>
➤ `ReactiveWebServerInitializedEvent` (WebFlux)
 Fired when embedded Tomcat/Jetty/Netty is ready.

### **9. DispatcherServlet Initialization (Spring MVC)**

Spring MVC configures:

* Handler mappings
* Interceptors
* Argument resolvers
* Message converters (JSON via Jackson)
* Exception handlers

Your REST endpoints are registered:

```java
@GetMapping("/hello")
public String hello() { return "Hello!"; }
```

### **10. Application Started Event Fired**

Spring publishes:

 ➤ `ApplicationStartedEvent`
 - ApplicationContext is refreshed. ✓
 - Web server is started. ✓
 - But CommandLineRunner and ApplicationRunner have not run yet ⟳
  
 ➤ `ApplicationReadyEvent`
 - Application is fully started, all runners executed. ✓
 - The app is ready to serve requests.  ✓
Any `@EventListener` for these will run.

### **11. Your Application Is Now Running**

The logs will show:

```
Started MyApp in 3.254 seconds (JVM running for 3.8)
```

Spring is now ready:

* Web server accepting requests
* Background scheduled tasks running
* Database connections established

### 12. Error Scenario 
➤ `ApplicationFailedEvent`

 
