Great question! This is a **common source of confusion**. Here's how authentication fits in:

---

### **OAuth vs Authentication**

- **OAuth** is purely for **authorization** — it tells a system **what the client is allowed to do**.
- **Authentication** is about **who the user is**.
- **OAuth delegates authentication to a trusted provider**, typically using **OpenID Connect (OIDC)**.

---

### **So, how does Authentication happen?**

If you want to **authenticate the user** (not just authorize access), you need:

### **OpenID Connect (OIDC) on top of OAuth 2.0**

- OIDC is an **identity layer** built on top of OAuth 2.0.
- It provides a **standard way to authenticate users** and get their **identity info** (via ID Token).

---

### **How it works with OIDC:**

1. **User logs in** to the Authorization Server (e.g., Keycloak, Auth0, Google).
2. The server returns:
   - **Access Token** (for authorization)
   - **ID Token (JWT)** — contains user identity info (used for authentication)
3. The client **verifies ID Token** and knows who the user is.

---

### Summary

| Purpose            | OAuth 2.0             | OpenID Connect (OIDC) |
|--------------------|-----------------------|------------------------|
| What it does       | Authorization         | Authentication + Auth |
| Token              | Access Token          | Access + ID Token     |
| Example use case   | API access control     | Login to an app       |

---

Would you like a Spring Boot example showing this flow?


Here's a **Spring Boot example** using **OAuth2 with OpenID Connect (OIDC)** for authentication and authorization via **Keycloak**:

---

### **Step 1: Setup Keycloak**

1. Run Keycloak via Docker:
   ```bash
   docker run -p 8080:8080 quay.io/keycloak/keycloak:24.0.2 start-dev
   ```

2. Open `http://localhost:8080/`
   - Create a realm (e.g., `myrealm`)
   - Create a client (e.g., `springboot-client`) with:
     - `Client ID`: `springboot-client`
     - `Root URL`: `http://localhost:8081`
     - Enable `Standard Flow`
     - Add a user and password under the Users section

---

### **Step 2: Spring Boot App Configuration**

#### `build.gradle` or `pom.xml`

**Gradle:**
```groovy
dependencies {
    implementation 'org.springframework.boot:spring-boot-starter-oauth2-client'
    implementation 'org.springframework.boot:spring-boot-starter-web'
}
```

---

#### `application.yml`

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
            client-secret: YOUR_CLIENT_SECRET
            authorization-grant-type: authorization_code
            redirect-uri: "{baseUrl}/login/oauth2/code/{registrationId}"
            scope: openid, profile, email
        provider:
          keycloak:
            issuer-uri: http://localhost:8080/realms/myrealm
```

---

### **Step 3: Secured Controller**

```java
@RestController
public class DemoController {

    @GetMapping("/secured")
    public String secured(@AuthenticationPrincipal OidcUser user) {
        return "Welcome, " + user.getFullName() + "! Email: " + user.getEmail();
    }

    @GetMapping("/")
    public String home() {
        return "Public Home Page";
    }
}
```

---

### **How it works:**

- Unauthenticated users visiting `/secured` are redirected to Keycloak login.
- Once authenticated, they are redirected back and authenticated using the **ID token**.

---

Would you like to see this with Spring Security filter breakdown or integrate it with multiple microservices?
