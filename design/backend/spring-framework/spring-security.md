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
