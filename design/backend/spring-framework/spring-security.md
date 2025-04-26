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


Good question!  
**Externalizing authentication in Spring Security** means **moving authentication logic outside** the Spring Boot application so that you don't hardcode users/passwords inside your application code (`inMemoryAuthentication`) but instead connect to **external systems** like:

| External Authentication | Example |
|:-------------------------|:--------|
| Database authentication  | JDBC / JPA-based authentication |
| LDAP authentication       | Active Directory or OpenLDAP |
| OAuth2 / OpenID Connect    | Google, Facebook, Okta |
| SSO (Single Sign-On)       | SAML 2.0, Keycloak |
| Custom Authentication Server | Your own API server |

---

### **Ways to Externalize Authentication**

1. **Database Authentication** (Using JDBC or JPA)
2. **LDAP Authentication** (Corporate users)
3. **OAuth2 / SSO Authentication** (Google, Facebook, SAML, Keycloak)
4. **Custom Authentication Provider** (Call external REST APIs)

---

### **1. Database Authentication Example (JDBC)**

```java
@Configuration
@EnableWebSecurity
public class SecurityConfig {

    @Autowired
    private DataSource dataSource; // Inject your database connection

    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        http
            .authorizeHttpRequests(auth -> auth.anyRequest().authenticated())
            .formLogin(Customizer.withDefaults())
            .csrf(AbstractHttpConfigurer::disable);

        return http.build();
    }

    @Bean
    public UserDetailsService userDetailsService() {
        JdbcUserDetailsManager manager = new JdbcUserDetailsManager(dataSource);
        return manager;
    }
}
```
- **Tables**: You should have standard `users` and `authorities` tables (Spring provides DDL).

---

### **2. LDAP Authentication Example**

```java
@Configuration
@EnableWebSecurity
public class SecurityConfig {

    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        http
            .authorizeHttpRequests(auth -> auth.anyRequest().authenticated())
            .formLogin(Customizer.withDefaults())
            .csrf(AbstractHttpConfigurer::disable)
            .ldapAuthentication(auth -> auth
                .userDnPatterns("uid={0},ou=people")
                .groupSearchBase("ou=groups")
                .contextSource()
                    .url("ldap://localhost:8389/dc=springframework,dc=org")
            );

        return http.build();
    }
}
```
- Authenticates against an **LDAP server** instead of DB.

---

### **3. OAuth2 / OpenID Connect Authentication**

Spring Boot supports **OAuth2 Login** easily:

```yaml
# application.yml
spring:
  security:
    oauth2:
      client:
        registration:
          google:
            client-id: YOUR_CLIENT_ID
            client-secret: YOUR_CLIENT_SECRET
            scope:
              - email
              - profile
```

```java
@Configuration
@EnableWebSecurity
public class SecurityConfig {

    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        http
            .authorizeHttpRequests(auth -> auth.anyRequest().authenticated())
            .oauth2Login(Customizer.withDefaults());
        return http.build();
    }
}
```
- Redirects to Google login, fetches token, and authenticates users.

---

### **4. Custom Authentication Provider (Custom REST API Call)**

If you have your own **authentication microservice**, you can build a **custom AuthenticationProvider**:

```java
@Component
public class CustomAuthenticationProvider implements AuthenticationProvider {

    @Override
    public Authentication authenticate(Authentication authentication) throws AuthenticationException {
        String username = authentication.getName();
        String password = (String) authentication.getCredentials();

        // Call your external authentication service here (REST API call)
        boolean isAuthenticated = externalService.authenticate(username, password);

        if (isAuthenticated) {
            return new UsernamePasswordAuthenticationToken(username, password, List.of(new SimpleGrantedAuthority("ROLE_USER")));
        } else {
            throw new BadCredentialsException("Authentication failed");
        }
    }

    @Override
    public boolean supports(Class<?> authentication) {
        return authentication.equals(UsernamePasswordAuthenticationToken.class);
    }
}
```

**Register this provider** in your security config.

---

### **Design Patterns Involved**

| Pattern | How it comes in |
|:--------|:----------------|
| **Strategy Pattern** | Different authentication mechanisms plugged dynamically |
| **Template Method** | HttpSecurity provides hooks for overriding behaviors |
| **Factory Pattern** | Beans like `AuthenticationManager`, `SecurityFilterChain` are created by factories |

---

### **Summary**

| External Authentication | When to use |
|:-------------------------|:------------|
| JDBC / JPA | Users stored in your database |
| LDAP | Enterprise users (corporate) |
| OAuth2 / SSO | Third-party login (Google, SAML) |
| Custom Provider | REST APIs or custom auth logic |

---

Would you like me to show a real-world example where Spring Security **delegates to an external REST API** for login? (It’s super common in microservices setups!)  
**(Yes/No?)**


Perfect!  
Let’s now build a **real-world Spring Boot example** where **Spring Security** authenticates users by **calling an external REST API** (e.g., an Authentication Server).

---

# **Spring Security Authentication via External REST API**

### **Overview**

- User submits login credentials
- Spring Security intercepts the request
- Instead of validating username/password locally (DB/LDAP),  
  we **call an external REST API** like:  
  `POST /auth/validate { username, password }`
- If external API says OK → Login success  
- If external API says Fail → Throw BadCredentialsException

---

# **Steps**

| Step | Description |
|:-----|:------------|
| 1    | Create a `CustomAuthenticationProvider` |
| 2    | Integrate it into `AuthenticationManager` |
| 3    | Configure Security FilterChain |
| 4    | Call external REST API inside custom provider |

---

# **1. CustomAuthenticationProvider**

```java
@Component
public class ExternalApiAuthenticationProvider implements AuthenticationProvider {

    private final RestTemplate restTemplate;

    public ExternalApiAuthenticationProvider() {
        this.restTemplate = new RestTemplate();
    }

    @Override
    public Authentication authenticate(Authentication authentication) throws AuthenticationException {
        String username = authentication.getName();
        String password = (String) authentication.getCredentials();

        // Create request payload
        Map<String, String> request = new HashMap<>();
        request.put("username", username);
        request.put("password", password);

        try {
            // Make REST call to external auth server
            ResponseEntity<Boolean> response = restTemplate.postForEntity(
                "http://localhost:8081/auth/validate",  // External API endpoint
                request,
                Boolean.class
            );

            if (Boolean.TRUE.equals(response.getBody())) {
                // Authentication successful
                List<GrantedAuthority> authorities = List.of(new SimpleGrantedAuthority("ROLE_USER"));
                return new UsernamePasswordAuthenticationToken(username, password, authorities);
            } else {
                throw new BadCredentialsException("External authentication failed");
            }
        } catch (Exception e) {
            throw new AuthenticationServiceException("Authentication server not reachable", e);
        }
    }

    @Override
    public boolean supports(Class<?> authentication) {
        return UsernamePasswordAuthenticationToken.class.isAssignableFrom(authentication);
    }
}
```

---

# **2. Security Configuration**

```java
@Configuration
@EnableWebSecurity
public class SecurityConfig {

    @Autowired
    private ExternalApiAuthenticationProvider externalAuthProvider;

    @Bean
    public AuthenticationManager authenticationManager(AuthenticationConfiguration config) throws Exception {
        return config.getAuthenticationManager();
    }

    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        http
            .authenticationProvider(externalAuthProvider) // Important!
            .authorizeHttpRequests(auth -> auth.anyRequest().authenticated())
            .formLogin(Customizer.withDefaults())
            .csrf(AbstractHttpConfigurer::disable);

        return http.build();
    }
}
```

---

# **3. External Authentication Server (Mock Example)**

You can mock the external authentication service separately.

```java
@RestController
@RequestMapping("/auth")
public class AuthController {

    @PostMapping("/validate")
    public ResponseEntity<Boolean> validate(@RequestBody Map<String, String> request) {
        String username = request.get("username");
        String password = request.get("password");

        // Dummy check (you can replace with real DB check here)
        if ("user".equals(username) && "password".equals(password)) {
            return ResponseEntity.ok(true);
        }
        return ResponseEntity.ok(false);
    }
}
```

_Run this on another server (say port 8081)._

---

# **Key Concepts Applied**

| Concept | Explanation |
|:--------|:------------|
| **AuthenticationProvider** | Customized login mechanism |
| **RestTemplate** | REST call to external server |
| **Spring Security Hooks** | Plugging custom provider |
| **Strategy Design Pattern** | Authentication strategies injected dynamically |
| **Separation of Concerns** | Auth server separated from application server |

---

# **Use Cases**

- **Microservices Authentication**: Call central auth server.
- **External API Verification**: e.g., OAuth servers, Auth0, Okta.
- **Centralized User Management**: Single place to manage users.
- **Legacy System Integration**: External systems handling credentials.

---

# **Diagram**

```
Browser --> Spring Boot App --> Calls --> External Auth API
          <-- Authentication Success / Failure
```

---

# **Would you like me to also show how to make this external call asynchronous using WebClient instead of RestTemplate?**
(For better performance)  
**(Yes/No?)**  
Awesome — let's now **integrate Spring Boot authentication with Keycloak**!  
(Keycloak = Open Source Identity Provider that supports OAuth2, OpenID Connect, SAML.)

---

# **Spring Boot + Keycloak Integration**

There are two ways:

| Mode | Explanation |
|:-----|:------------|
| 1. OAuth2 Login (Recommended) | Use Spring Security's built-in OAuth2 login, Keycloak as the IdP (Identity Provider). |
| 2. Keycloak Adapter (Deprecated) | Old way (with `keycloak-spring-boot-starter`) — no longer recommended. |

We will use **Method 1 (OAuth2 Login with Spring Security 6)** — *modern, simple, official way.*

---

# **High Level Flow**

1. User clicks login.
2. Redirects to Keycloak login page.
3. Enters credentials → Keycloak authenticates.
4. Keycloak redirects back to Spring Boot app with a Token.
5. Spring Boot uses the token to allow or reject access.

---

# **Steps to Integrate**

| Step | Details |
|:-----|:--------|
| 1 | Set up Keycloak server |
| 2 | Create a Realm, Client, User in Keycloak |
| 3 | Configure Spring Boot application to use OAuth2 client |
| 4 | Secure Spring Boot application using Spring Security |

---

# **1. Setup Keycloak**

- Download Keycloak server: https://www.keycloak.org/downloads
- Run Keycloak:

```bash
bin/kc.sh start-dev
```
_Default Admin Console at_ → `http://localhost:8080/admin`

- Create **Realm**: `myrealm`
- Create **Client**:
  - Client ID: `springboot-client`
  - Client Protocol: `openid-connect`
  - Access Type: `confidential`
  - Valid Redirect URIs: `http://localhost:8081/login/oauth2/code/*`
  - Client Authentication: ON
- Create **User**:
  - Username: `testuser`
  - Password: `password`
  - Assign roles if needed.

---

# **2. Spring Boot Configuration**

Add Maven Dependencies:

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-oauth2-client</artifactId>
</dependency>
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-security</artifactId>
</dependency>
```

---

# **3. `application.yml`**

```yaml
server:
  port: 8081

spring:
  security:
    oauth2:
      client:
        registration:
          keycloak:
            client-id: springboot-client
            client-secret: YOUR-CLIENT-SECRET
            scope: openid
            redirect-uri: "{baseUrl}/login/oauth2/code/{registrationId}"
            authorization-grant-type: authorization_code
        provider:
          keycloak:
            issuer-uri: http://localhost:8080/realms/myrealm
            user-name-attribute: preferred_username
```

_(Replace `YOUR-CLIENT-SECRET` from Keycloak Client settings.)_

---

# **4. Security Configuration**

```java
@Configuration
@EnableWebSecurity
public class SecurityConfig {

    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        http
            .authorizeHttpRequests(auth -> auth
                .anyRequest().authenticated()
            )
            .oauth2Login(Customizer.withDefaults()); // Enable OAuth2 login

        return http.build();
    }
}
```

---

# **Now Start the Spring Boot app!**

1. Visit: `http://localhost:8081`
2. It will **redirect to Keycloak login page**.
3. Login with `testuser/password`
4. Successfully authenticated → redirected back to the app.

---

# **Quick Diagram**

```
Browser --> Spring Boot App --> Redirect --> Keycloak Login
Browser <-- Redirect <-- Successful Token
Access Protected Resources
```

---

# **Bonus: Access Token Details**

If you want to see **user details** after login:

```java
@RestController
public class HomeController {

    @GetMapping("/")
    public String home(@AuthenticationPrincipal OidcUser oidcUser) {
        return "Hello " + oidcUser.getFullName();
    }
}
```
- You can fetch Email, Roles, Name, etc., from the Keycloak Token.

---

# **Design Patterns Used**

| Pattern | Explanation |
|:--------|:------------|
| **OAuth2 Authorization Code Flow** | Secure login protocol |
| **Template Method** | Spring Security extends OAuth2LoginConfigurer |
| **Strategy Pattern** | Pluggable authentication providers (Keycloak) |

---

# **Use Cases**

- Centralized authentication across microservices
- Role-based access control via Keycloak groups
- OAuth2 / OpenID login for applications

---

# **Would you also like me to show how to secure specific endpoints using Keycloak roles (RBAC)?**  
_(e.g., only admin can access `/admin`, users access `/user`)_

**(Yes/No?)**  
It’s super useful in real-world apps!


Awesome!  
Let’s now **extend Keycloak integration** to **secure specific endpoints based on user roles** (Role-Based Access Control - RBAC).

---

# **Securing Specific Endpoints with Keycloak Roles in Spring Boot**

---

# **1. Keycloak Setup: Add Roles to Users**

- Go to Keycloak Admin Console → `myrealm`
- Create Roles:
  - `ROLE_USER`
  - `ROLE_ADMIN`
- Assign roles to users:
  - `testuser` → Assign `ROLE_USER`
  - Create another user `adminuser` → Assign `ROLE_ADMIN`

**Important**:  
Spring Security expects roles to start with **`ROLE_`** prefix by default.

---

# **2. Customize Spring Security Configuration**

We need to tell Spring Boot:
- How to **map Keycloak roles** correctly
- Secure different URL patterns based on roles

---

# **Updated `SecurityConfig.java`**

```java
@Configuration
@EnableWebSecurity
public class SecurityConfig {

    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        http
            .authorizeHttpRequests(auth -> auth
                .requestMatchers("/admin/**").hasRole("ADMIN")  // Only ADMIN role
                .requestMatchers("/user/**").hasRole("USER")    // Only USER role
                .anyRequest().authenticated()                  // Other requests must be authenticated
            )
            .oauth2Login(Customizer.withDefaults())  // OAuth2 login
            .oauth2ResourceServer(oauth2 -> oauth2.jwt()); // (optional if using Bearer tokens)

        return http.build();
    }

    @Bean
    public GrantedAuthoritiesMapper authoritiesMapper() {
        return authorities -> authorities.stream()
            .map(authority -> {
                String roleName = authority.getAuthority();
                if (roleName.startsWith("ROLE_")) {
                    return authority;
                }
                return new SimpleGrantedAuthority("ROLE_" + roleName);
            })
            .collect(Collectors.toSet());
    }
}
```

---

# **Explanation**

| Feature | Purpose |
|:--------|:--------|
| `requestMatchers("/admin/**").hasRole("ADMIN")` | Only allow users with `ROLE_ADMIN` |
| `requestMatchers("/user/**").hasRole("USER")` | Only allow users with `ROLE_USER` |
| `GrantedAuthoritiesMapper` | Automatically add `ROLE_` prefix if missing (Keycloak usually sends raw roles without prefix) |

---

# **3. Controller Example**

```java
@RestController
@RequestMapping("/")
public class HomeController {

    @GetMapping
    public String home() {
        return "Public Home Page - No Authentication Required.";
    }

    @GetMapping("/user/dashboard")
    public String userDashboard(@AuthenticationPrincipal OidcUser oidcUser) {
        return "Welcome USER: " + oidcUser.getFullName();
    }

    @GetMapping("/admin/dashboard")
    public String adminDashboard(@AuthenticationPrincipal OidcUser oidcUser) {
        return "Welcome ADMIN: " + oidcUser.getFullName();
    }
}
```

---

# **Behavior Now**

| URL | Who Can Access |
|:----|:---------------|
| `/` | Everyone |
| `/user/dashboard` | Only users with `ROLE_USER` |
| `/admin/dashboard` | Only users with `ROLE_ADMIN` |

If you access without proper role → **403 Forbidden**.

---

# **Bonus Tip:**  
If you want fine-grained method security:

```java
@EnableMethodSecurity
public class SecurityConfig { ... }
```

Then you can put annotations like:

```java
@PreAuthorize("hasRole('ADMIN')")
@GetMapping("/admin/dashboard")
public String adminDashboard(...) { ... }
```

**(Very useful for microservices!)**

---

# **Design Patterns Used Here**

| Pattern | Purpose |
|:--------|:--------|
| **Strategy** | Different authorization strategies based on roles |
| **Adapter** | Mapping Keycloak roles to Spring Security authorities |
| **Proxy (Method Security)** | Spring creates proxy to check access before method execution |

---

# **Quick Visual**

```
Browser → /user/dashboard → Allowed only if ROLE_USER
Browser → /admin/dashboard → Allowed only if ROLE_ADMIN
```

Keycloak → Sends roles → Spring Security → Checks roles → Grants/Deny access.

---

# **Summary**

- Configure Keycloak to have user roles
- Map Keycloak roles correctly in Spring
- Secure different endpoints based on roles
- Bonus: use method-level security too!

---

# **Would you also like me to show how to handle role mapping automatically with Keycloak’s built-in Group-to-Role mapping?**
(Really useful when users belong to Groups like “Admin Group” → auto assigned `ROLE_ADMIN`.)  
**(Yes/No?)**  
It makes managing 1000s of users super easy!