# **Spring Security: Concepts and Working Examples**  

Spring Security is a **powerful authentication and authorization framework** used to secure Spring applications. It provides features like **user authentication, role-based access control (RBAC), OAuth2, JWT authentication, CSRF protection, and method-level security**.

---

# **🔹 1. Key Concepts in Spring Security**
| Concept | Description |
|---------|------------|
| **Authentication** | Verifying user identity (e.g., username & password). |
| **Authorization** | Determining user permissions (e.g., role-based access). |
| **Filter Chain** | A series of security filters applied before requests reach controllers. |
| **UserDetailsService** | Custom user authentication service. |
| **SecurityContext** | Stores authentication details per session/request. |
| **JWT (JSON Web Token)** | Token-based authentication mechanism. |
| **OAuth2** | Authentication using third-party providers (Google, Facebook, etc.). |

---

# **🔹 2. Adding Spring Security to a Spring Boot Project**
### **✅ Step 1: Add Dependencies**
```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-security</artifactId>
</dependency>

<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-web</artifactId>
</dependency>

<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-data-jpa</artifactId>
</dependency>

<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>org.springframework.boot-starter-thymeleaf</artifactId>
</dependency>

<dependency>
    <groupId>com.h2database</groupId>
    <artifactId>h2</artifactId>
    <scope>runtime</scope>
</dependency>
```
🔹 **Spring Boot automatically applies security, requiring login for all endpoints.**

---

# **🔹 3. Default Security Setup**
Spring Security automatically configures a **basic authentication** system:
- **Default username** → `user`
- **Password** → Generated in logs (on startup)
- **Default login page** → `/login`

```log
.UserDetailsServiceAutoConfiguration : 

Using generated security password:
```
By Default all requests return 401
```bash
curl -i http://localhost:8080/api/employees -H "content-type:application/json" -d '{"employeeName":"Rahul" , "salary": 2000}'
```
```log
HTTP/1.1 401 

```
To make the GET APIs working you can use the password generated in console ,below is the curl command
```bash
curl -i   -u user:$pass http://localhost:8080/api/employees
```
The POST endpoints will still throw 401 , To fix the POST endpoints you need to make code changes 

 
This happens because of **CSRF protection**!

---
- By default, **Spring Security enables CSRF protection** for **stateful sessions** (i.e., browser clients).
- When you send a **POST/PUT/DELETE**, Spring Security **expects a CSRF token**.
- Since your `curl` or API client is not sending any CSRF token, Spring rejects the request with 401 or 403.

---

###  Solution options:

| Option | What to do | When to use |
|:---|:---|:---|
| 1. Disable CSRF protection (easy for APIs) | Add `http.csrf().disable()` in security config | For **pure APIs** (no browser form login) |
| 2. Send CSRF token manually | Use Spring's CSRF endpoint to get token first | Needed if you are using browser + forms |

**Since you are building an API, option 1 (disable CSRF) is perfectly fine!**

---

###  Quick Code to disable CSRF:

Create a `SecurityConfig.java` if you don’t have it:

```java
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.web.SecurityFilterChain;

@Configuration
public class SecurityConfig {

    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        http
            .csrf().disable() //   Disable CSRF for APIs
            .authorizeHttpRequests(authorize -> authorize
                .anyRequest().authenticated()
            )
            .httpBasic(); // Use basic auth
        return http.build();
    }
}
```

---

### 🧪 After this, your POST cURL should work like:

```bash
curl -u user:$pass -H "Content-Type: application/json" -X POST -d '{"key":"value"}' http://localhost:8080/your-post-endpoint
```

✅ No CSRF token needed anymore!
 

---

# **🔹 4. Custom Security Configuration**
By default, Spring Security **protects all endpoints**. To customize security:

### **✅ Step 1: Create a Custom Security Configuration**
```java
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.core.userdetails.User;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.provisioning.InMemoryUserDetailsManager;
import org.springframework.security.web.SecurityFilterChain;

@Configuration
@EnableWebSecurity
/**
* for spring security setup using Configuration and EnableWebSecurity to create SecurityFilterChain bean for  
*/
public class SecurityConfig {
    
    @Bean
    public SecurityFilterChain securityFilterChain(HttpSecurity http) throws Exception {
        http
            .authorizeHttpRequests(auth -> auth
                .requestMatchers("/admin").hasRole("ADMIN") // Only ADMIN can access
                .requestMatchers("/user").hasAnyRole("USER", "ADMIN") // USER & ADMIN can access
                .anyRequest().authenticated() // Other requests require authentication
            )
            .formLogin(form -> form // Enable form-based login
                .defaultSuccessUrl("/home", true)
                .permitAll()
            )
            .logout(logout -> logout.logoutSuccessUrl("/login?logout").permitAll());

        return http.build();
    }

    @Bean
    public UserDetailsService userDetailsService() {
        UserDetails user = User.withDefaultPasswordEncoder()
                .username("user")
                .password("password")
                .roles("USER")
                .build();

        UserDetails admin = User.withDefaultPasswordEncoder()
                .username("admin")
                .password("admin123")
                .roles("ADMIN")
                .build();

        return new InMemoryUserDetailsManager(user, admin);
    }
}
```
### **Explanation:**
✔ **Defines access rules:**  
   - `/admin` → Only accessible by **ADMIN**  
   - `/user` → Accessible by **USER & ADMIN**  
   - **Other URLs** → Require authentication  
✔ **Enables form-based login** instead of Basic Authentication  
✔ **Defines two users** in memory (`user` and `admin`)  

---

# **🔹 5. Database Authentication with JPA**
Instead of in-memory users, we can store user credentials in a database.

### **✅ Step 1: Define User Entity**
```java
import jakarta.persistence.*;
import java.util.Set;

@Entity
@Table(name = "users")
public class User {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    
    private String username;
    private String password;
    private String role; // ROLE_USER, ROLE_ADMIN

    // Getters & Setters
}
```

### **✅ Step 2: Create Repository**
```java
import org.springframework.data.jpa.repository.JpaRepository;
import java.util.Optional;

public interface UserRepository extends JpaRepository<User, Long> {
    Optional<User> findByUsername(String username);
}
```

### **✅ Step 3: Implement UserDetailsService**
```java
import org.springframework.security.core.userdetails.*;
import org.springframework.stereotype.Service;
import java.util.Optional;

@Service
public class CustomUserDetailsService implements UserDetailsService {
    private final UserRepository userRepository;

    public CustomUserDetailsService(UserRepository userRepository) {
        this.userRepository = userRepository;
    }

    @Override
    public UserDetails loadUserByUsername(String username) throws UsernameNotFoundException {
        User user = userRepository.findByUsername(username)
                .orElseThrow(() -> new UsernameNotFoundException("User not found"));

        return User.withUsername(user.getUsername())
                .password(user.getPassword())
                .roles(user.getRole()) // Convert role to Spring Security format
                .build();
    }
}
```

### **✅ Step 4: Use Database Authentication in Security Config**
```java
@Bean
public UserDetailsService userDetailsService(UserRepository userRepository) {
    return new CustomUserDetailsService(userRepository);
}
```
🔹 **Now, user credentials will be validated from the database.**

---

# **🔹 6. JWT Authentication**
Spring Security also supports **JWT-based authentication**.

### **How JWT Works?**
1. User logs in and receives a **JWT token**.
2. The client includes this token in **Authorization: Bearer <token>**.
3. Spring Security validates the token and grants access.

---

# **🔹 7. Method-Level Security**
Spring Security can restrict access at the **method level** using annotations.

### **✅ Enable Method Security**
```java
@EnableMethodSecurity
@Configuration
public class SecurityConfig {
    ...
}
```

### **✅ Secure Controller Methods**
```java
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api")
public class UserController {

    @GetMapping("/user")
    @PreAuthorize("hasRole('USER') or hasRole('ADMIN')")
    public String userAccess() {
        return "User Content";
    }

    @GetMapping("/admin")
    @PreAuthorize("hasRole('ADMIN')")
    public String adminAccess() {
        return "Admin Content";
    }
}
```
🔹 **Restricts access at the method level using roles.**

---

# **🔹 8. CSRF Protection**
Spring Security **enables CSRF protection by default** to prevent attacks.

### **✅ Disable CSRF for APIs (Not Recommended for Forms)**
```java
http.csrf(csrf -> csrf.disable());
```
🔹 **Use for stateless REST APIs** where CSRF is unnecessary.

---

# **🔹 9. OAuth2 Authentication**
Spring Security supports **OAuth2 login** with providers like **Google, Facebook, GitHub**.

### **✅ Add OAuth2 Dependencies**
```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-oauth2-client</artifactId>
</dependency>
```

### **✅ Enable OAuth2 Login**
```java
http.oauth2Login(Customizer.withDefaults());
```
🔹 **Now users can log in using Google, Facebook, or GitHub.**

---

# **🔹 Summary**
| Feature | Description |
|---------|------------|
| **Authentication** | Verifies users via DB, JWT, or OAuth2. |
| **Authorization** | Grants/restricts access based on roles. |
| **Method-Level Security** | Restricts specific methods using `@PreAuthorize`. |
| **JWT Support** | Allows token-based authentication. |
| **OAuth2** | Enables login with Google, Facebook, etc. |

Would you like a complete JWT-based authentication example? 🚀



Got it — you're hitting the **default Spring Security behavior** after adding the `spring-boot-starter-security` dependency.

Here’s exactly what's happening:

---

### 🔥 Why is a random password being printed?

When you **add** `spring-boot-starter-security`, and you **don't configure** any custom security settings, Spring Boot **auto-configures** basic HTTP authentication:

- It creates a **default user** with username `user`.
- It generates a **random password** at startup (you can see it in logs — something like this):

```text
Using generated security password: 8f8c61dd-67a3-4b5f-80e2-75db8aaf3b55
```

This is **intended to secure** the application immediately.

---

### 🚨 Why are your API calls failing with 401 Unauthorized?

Because now **every API** requires authentication (username/password) unless you explicitly configure it otherwise.

Since your client (Postman, curl, frontend, etc.) isn't providing credentials, **Spring Security** blocks the request with **HTTP 401 Unauthorized**.

---

### 🛠 How to Fix It?

You have 2 options:

---

#### 1. (Quick fix) **Allow all APIs without authentication** (for development)

If you don't want security at all (during dev), create a `SecurityConfig` class like this:

```java
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.web.SecurityFilterChain;

@Configuration
public class SecurityConfig {

    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        http
            .csrf().disable() // (optional) disable CSRF for POST/PUT APIs in development
            .authorizeHttpRequests(authorize -> authorize
                .anyRequest().permitAll()
            );
        return http.build();
    }
}
```

✅ This will **allow all requests** without authentication.

---

#### 2. (Better) **Configure proper authentication**

If you actually want to **secure APIs properly** (and not just turn it off), you should:

- Define users, roles, permissions.
- Configure login endpoints or token-based authentication (e.g., JWT).

Example: In-memory username/password:

```java
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.core.userdetails.User;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.provisioning.InMemoryUserDetailsManager;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;

@Configuration
public class SecurityConfig {

    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        http
            .csrf().disable()
            .authorizeHttpRequests(authz -> authz
                .anyRequest().authenticated()
            )
            .httpBasic();
        return http.build();
    }

    @Bean
    public UserDetailsService users() {
        return new InMemoryUserDetailsManager(
            User.withUsername("admin")
                .password(passwordEncoder().encode("admin123"))
                .roles("USER")
                .build()
        );
    }

    @Bean
    public PasswordEncoder passwordEncoder() {
        return new BCryptPasswordEncoder();
    }
}
```

Now you can call APIs by using:
- Username: `admin`
- Password: `admin123`

---

### ⚡️ Summary

| Situation            | Action                            |
|-----------------------|-----------------------------------|
| Disable security (dev) | Use `.permitAll()` config |
| Add simple auth       | Setup `InMemoryUserDetailsManager` |
| Production app        | Setup JWT, OAuth2, etc. |

---

### 🔥 Quick Pro Tip
If you don't want to manually create `SecurityConfig` for now (only disable),  
**you can even remove the `spring-boot-starter-security` dependency** from your `pom.xml` until you really need it.  
Then APIs won't be protected.

---

Would you like me to show how to add **JWT-based authentication** next?  
(That's usually better than Basic Auth for real projects!) 🚀  
Want me to?
