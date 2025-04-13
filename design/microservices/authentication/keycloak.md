Here’s a clear explanation and working examples of **authentication** and **authorization** using:

- `client_credentials` (machine-to-machine)
- `password` grant (username/password)
- With **Spring Security + Keycloak**

---

## **1. Setup Keycloak**

1. **Create a realm**: `demo-realm`
2. **Create clients**:
   - `spring-client` with `client_credentials` and `password` flow enabled.
3. **Create a user**:
   - `username: user1`, `password: password`, roles: `USER`

---

## **2. Spring Boot Security Configuration (with Keycloak)**

### **Dependencies**

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-oauth2-resource-server</artifactId>
</dependency>
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-security</artifactId>
</dependency>
```

---

## **3. application.yml**

```yaml
spring:
  security:
    oauth2:
      resourceserver:
        jwt:
          issuer-uri: http://localhost:8080/realms/demo-realm
      client:
        registration:
          keycloak-client:
            client-id: spring-client
            client-secret: secret
            authorization-grant-type: client_credentials
            scope: openid
        provider:
          keycloak:
            issuer-uri: http://localhost:8080/realms/demo-realm
```

---

## **4. Secure Controller**

```java
@RestController
public class DemoController {

    @GetMapping("/public")
    public String publicEndpoint() {
        return "No Auth Needed";
    }

    @PreAuthorize("hasRole('USER')")
    @GetMapping("/user")
    public String userEndpoint() {
        return "Hello, Authenticated User";
    }
}
```

---

## **5. Client Credentials Example (machine to machine)**

Using **Spring WebClient** to call `/user`:

```java
@Component
public class MachineClient {

    private final WebClient client;

    public MachineClient(WebClient.Builder builder) {
        this.client = builder.build();
    }

    @PostConstruct
    public void callSecureEndpoint() {
        client.get()
              .uri("http://localhost:8081/user")
              .headers(h -> h.setBearerAuth(getToken()))
              .retrieve()
              .bodyToMono(String.class)
              .subscribe(System.out::println);
    }

    private String getToken() {
        return client.post()
                .uri("http://localhost:8080/realms/demo-realm/protocol/openid-connect/token")
                .contentType(MediaType.APPLICATION_FORM_URLENCODED)
                .bodyValue("grant_type=client_credentials&client_id=spring-client&client_secret=secret")
                .retrieve()
                .bodyToMono(JsonNode.class)
                .map(token -> token.get("access_token").asText())
                .block();
    }
}
```

---

## **6. Password Grant Example (username/password login)**

```java
private String getUserToken() {
    return client.post()
        .uri("http://localhost:8080/realms/demo-realm/protocol/openid-connect/token")
        .contentType(MediaType.APPLICATION_FORM_URLENCODED)
        .bodyValue("grant_type=password&username=user1&password=password&client_id=spring-client&client_secret=secret")
        .retrieve()
        .bodyToMono(JsonNode.class)
        .map(token -> token.get("access_token").asText())
        .block();
}
```

---

### **Authentication vs Authorization**

| Feature           | Authentication                                      | Authorization                       |
|-------------------|------------------------------------------------------|--------------------------------------|
| **Purpose**        | Who are you?                                         | What can you do?                     |
| **Keycloak Role**  | Validates user credentials                          | Provides roles/permissions           |
| **Spring Role**    | Uses JWT/Token to extract identity                  | Uses `@PreAuthorize`, role checks    |
| **Client Creds**   | For services with no users                          | Grants access via roles to machines  |
| **Password Grant** | For human users                                     | Role assigned in Keycloak per user   |

---

Would you like this as a runnable repo or Postman collection too?