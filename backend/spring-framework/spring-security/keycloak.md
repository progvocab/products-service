- use **Keycloak** with Spring Boot for token-based authentication, 
 it will result in **significantly less code** compared to writing your own JWT issuing and validating logic.

---

## ✅ Why Use Keycloak?

| Feature                    | DIY JWT Code        | Keycloak                |
| -------------------------- | ------------------- | ----------------------- |
| Secure token issuing       | You write it        | Provided out of the box |
| Token validation           | You code filters    | Built-in Spring support |
| Expiration, roles, scopes  | You manage manually | Handled automatically   |
| Revocation, refresh tokens | You implement       | Keycloak manages        |
| UI/Admin management        | N/A                 | Admin Console available |

So yes — **less code, more security, easier scaling**.

---

## ✅ How It Works with Spring Boot

### 🛠️ You just configure:

* `spring.security.oauth2.resourceserver.jwt.issuer-uri=https://<keycloak-server>/realms/<realm-name>`

Spring Boot + Spring Security + Keycloak handles the rest:

* Validates tokens
* Sets up security context
* Restricts endpoints by roles/authorities

---

## 🧰 Quick Setup Guide

### 1. **Run Keycloak**

Use Docker for a quick start:

```bash
docker run -p 8080:8080 \
  -e KEYCLOAK_ADMIN=admin \
  -e KEYCLOAK_ADMIN_PASSWORD=admin \
  quay.io/keycloak/keycloak:24.0.2 start-dev
```

Go to: `http://localhost:8080`

---

### 2. **Create Realm, Client & Role**

1. Create Realm: `myrealm`
2. Create Client:

   * ID: `myclient`
   * Client Authentication: ✅ Enabled
   * Access Type: `confidential` or `public` (depending on flow)
   * Valid Redirect URIs: `*` (for testing)
3. Create User: `client1` with password
4. Assign roles to the user (optional)

---

### 3. **Spring Boot `application.properties`**

```properties
spring.security.oauth2.resourceserver.jwt.issuer-uri=http://localhost:8080/realms/myrealm
spring.security.oauth2.resourceserver.jwt.jwk-set-uri=http://localhost:8080/realms/myrealm/protocol/openid-connect/certs
```

---

### 4. **Add Dependencies (`pom.xml`)**

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-oauth2-resource-server</artifactId>
</dependency>
```

---

### 5. **Secure Endpoints with Roles**

```java
@RestController
public class DemoController {

    @GetMapping("/public")
    public String publicApi() {
        return "Accessible to everyone";
    }

    @GetMapping("/secure")
    @PreAuthorize("hasAuthority('SCOPE_profile')") // Or just @PreAuthorize("isAuthenticated()")
    public String secureApi() {
        return "Authenticated access";
    }
}
```

Enable `@PreAuthorize`:

```java
@EnableGlobalMethodSecurity(prePostEnabled = true)
@Configuration
public class MethodSecurityConfig extends GlobalMethodSecurityConfiguration {}
```

---

### 6. **Test**

Use Postman or `curl`:

```bash
curl -X POST http://localhost:8080/realms/myrealm/protocol/openid-connect/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "client_id=myclient" \
  -d "username=client1" \
  -d "password=client1pass" \
  -d "grant_type=password"
```

Then call:

```bash
curl -H "Authorization: Bearer <access_token>" http://localhost:8081/secure
```

---

## 🔑 Summary

Yes, **Keycloak simplifies your setup**:

| Task                        | Without Keycloak | With Keycloak |
| --------------------------- | ---------------- | ------------- |
| Token generation/validation | Manual           | Automatic     |
| Securing endpoints          | Custom filters   | Spring config |
| Managing clients/users      | Code/properties  | Web UI/Admin  |
| Token revocation/refresh    | Manual           | Built-in      |

---

## ➕ Bonus

You can also use:

* `spring-boot-starter-oauth2-client` for frontend OAuth2 login
* Custom claims mapping to Spring roles
* Realm-level or client-specific roles

---

Would you like a complete working **Spring Boot + Keycloak example project** (starter template)?

To **validate a Keycloak JWT token** in a Spring Boot application, you'll typically do this using:

* Spring Security + OAuth2 Resource Server
* Keycloak public keys to verify the token signature
* Roles/claims to authorize requests

---

## ✅ 1. Basic Setup

### 📦 `pom.xml` dependencies

```xml
<dependencies>
    <!-- Spring Boot Security -->
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-security</artifactId>
    </dependency>

    <!-- Resource server to decode and validate JWT -->
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-oauth2-resource-server</artifactId>
    </dependency>

    <!-- JWT decoder -->
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-oauth2-jose</artifactId>
    </dependency>
</dependencies>
```

---

## ⚙️ 2. `application.properties`

```properties
# Replace with your Keycloak realm URL
spring.security.oauth2.resourceserver.jwt.issuer-uri=http://localhost:8080/realms/myrealm

# Optional (auto-discovered from issuer-uri): 
# spring.security.oauth2.resourceserver.jwt.jwk-set-uri=http://localhost:8080/realms/myrealm/protocol/openid-connect/certs
```

This will:

* Fetch the **public key (JWK)** from Keycloak
* Use it to **validate signature** of the JWT
* Validate token **expiration**, **issuer**, and **audience** (if configured)

---

## 🔐 3. Enable Resource Server

```java
package com.example.security;

import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.context.annotation.Bean;

@Configuration
@EnableWebSecurity
public class SecurityConfig {

    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        http
            .authorizeHttpRequests(auth -> auth
                .requestMatchers("/public/**").permitAll()
                .anyRequest().authenticated()
            )
            .oauth2ResourceServer(oauth2 -> oauth2
                .jwt()  // Enable JWT token validation
            );
        return http.build();
    }
}
```

---

## 📥 4. Use Bearer Token in Requests

When calling your secured endpoints:

```
GET /api/employees
Authorization: Bearer <access_token>
```

---

## 🧠 5. Decode JWT Manually (Optional)

If you want to read token claims in code:

```java
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.oauth2.jwt.Jwt;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api")
public class MyController {

    @GetMapping("/userinfo")
    public String userInfo(@AuthenticationPrincipal Jwt jwt) {
        String username = jwt.getClaim("preferred_username");
        return "Hello " + username;
    }
}
```

---

## ✅ Validated By Spring

Once configured, Spring automatically:

* Validates JWT structure and signature
* Validates expiration and issuer
* Injects token claims into `AuthenticationPrincipal` or `SecurityContextHolder`

---

## 🔧 Keycloak Notes

* Be sure the **audience** (`aud`) in the token matches your client.
* If using client credentials, enable “Service Accounts” in Keycloak client settings.


### 🔐 What Are **Keycloak Public Keys**?

Keycloak uses **asymmetric cryptography** (usually **RSA**) to sign the **JWT tokens** it issues. These tokens are signed with a **private key** that only Keycloak holds, and can be **verified by clients** (like your Spring Boot app) using a **public key**.

---

### 🔑 Key Concepts

| Key Type        | Description                                                           |
| --------------- | --------------------------------------------------------------------- |
| **Private Key** | Used by Keycloak to sign JWT tokens                                   |
| **Public Key**  | Used by applications to verify the token's authenticity and integrity |

---

### 📍 Where to Find Keycloak’s Public Key?

There are 2 ways to get the public key:

#### ✅ 1. Via OpenID Connect Discovery URL (Recommended)

```http
GET http://<keycloak-host>/realms/<realm-name>/.well-known/openid-configuration
```

Example:

```
http://localhost:8080/realms/myrealm/.well-known/openid-configuration
```

From the JSON, look for:

```json
"jwks_uri": "http://localhost:8080/realms/myrealm/protocol/openid-connect/certs"
```

This `jwks_uri` contains the **JSON Web Key Set (JWKS)** — the public keys in JWK format.

#### ✅ 2. Admin Console (Manual)

1. Go to **Realm Settings > Keys**
2. Click on the **Active RSA key**
3. Copy the **Public Key**

Example:

```text
MIIBIjANBgkqhkiG9w0BAQEFAA...
```

You can embed this in your Spring config (not recommended if you rotate keys).

---

### 📦 How Spring Boot Uses the Public Key

When you set:

```properties
spring.security.oauth2.resourceserver.jwt.issuer-uri=http://localhost:8080/realms/myrealm
```

Spring Boot automatically:

* Reads the discovery document
* Fetches the JWKS (`jwks_uri`)
* Caches the public key
* Validates every incoming token's signature with that public key

---

### 🔍 Example: Public Key in JWKS Format

```json
{
  "keys": [
    {
      "kid": "abc123",
      "kty": "RSA",
      "alg": "RS256",
      "use": "sig",
      "n": "0vx7agoebGc... (modulus)",
      "e": "AQAB"
    }
  ]
}
```

Spring Boot automatically uses this to verify tokens signed with `RS256`.

---

### ✅ Summary

| Feature            | Value                                            |
| ------------------ | ------------------------------------------------ |
| Public key purpose | Verify JWTs signed by Keycloak                   |
| Where to find it   | `/.well-known/openid-configuration` → `jwks_uri` |
| Used by            | Spring Boot, API Gateways, any JWT validator     |
| Key format         | JWK (JSON Web Key)                               |
| Key algorithm      | RS256 (default), EC supported                    |

---

Would you like help decoding your Keycloak JWT and verifying the signature using the public key?


When using Keycloak in a Spring Boot project (especially with Spring Security and the OAuth2 Resource Server), **Keycloak sets up the `SecurityContext` automatically** using the JWT token provided by the client.

Let’s break this down step-by-step to understand how the **SecurityContext is built** and how your application can use it.

---

## 🔐 Keycloak + Spring Boot Security Context Setup

### 🔁 Flow Overview

1. **Client** sends a request with a `Bearer` token (JWT) from Keycloak.
2. **Spring Security's OAuth2ResourceServer**:

   * Validates the token (signature, expiration, issuer).
   * Converts token claims to a `JwtAuthenticationToken`.
   * Stores this in the **`SecurityContextHolder`**.
3. You can access the **current authenticated user** using:

   * `@AuthenticationPrincipal Jwt`
   * `SecurityContextHolder.getContext().getAuthentication()`

---

## ⚙️ Spring Boot Configuration

### 1. `application.properties`

```properties
spring.security.oauth2.resourceserver.jwt.issuer-uri=http://localhost:8080/realms/myrealm
```

This:

* Automatically fetches Keycloak's public key (from the JWKS URI)
* Enables JWT signature validation
* Builds `JwtAuthenticationToken`

---

### 2. Security Config (`SecurityFilterChain`)

```java
package com.example.security;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.web.SecurityFilterChain;

@Configuration
public class SecurityConfig {

    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        http
            .authorizeHttpRequests(auth -> auth
                .requestMatchers("/public/**").permitAll()
                .anyRequest().authenticated()
            )
            .oauth2ResourceServer(oauth -> oauth
                .jwt() // enables SecurityContext setup
            );

        return http.build();
    }
}
```

---

## ✅ SecurityContext Contents

After a token is successfully validated, Spring Security creates:

```java
JwtAuthenticationToken token = new JwtAuthenticationToken(jwt);
SecurityContextHolder.getContext().setAuthentication(token);
```

### Example Claims in Token:

```json
{
  "sub": "user-id-123",
  "preferred_username": "alice",
  "email": "alice@example.com",
  "realm_access": {
    "roles": ["USER", "ADMIN"]
  }
}
```

---

## 🧠 How to Access the SecurityContext in Code

### Method 1: Use `@AuthenticationPrincipal`

```java
@RestController
public class UserController {

    @GetMapping("/me")
    public String getMe(@AuthenticationPrincipal Jwt jwt) {
        return "Logged in as: " + jwt.getClaimAsString("preferred_username");
    }
}
```

### Method 2: Use `SecurityContextHolder`

```java
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.oauth2.jwt.Jwt;
import org.springframework.security.authentication.AbstractAuthenticationToken;

public String getCurrentUser() {
    AbstractAuthenticationToken auth = (AbstractAuthenticationToken) SecurityContextHolder.getContext().getAuthentication();
    Jwt jwt = (Jwt) auth.getPrincipal();
    return jwt.getClaimAsString("preferred_username");
}
```

---

## 🎯 Optional: Map Keycloak Roles to Spring Authorities

Spring won’t use Keycloak roles (`realm_access.roles`) as authorities by default.

### Add this to map them:

```java
import org.springframework.security.oauth2.server.resource.authentication.JwtAuthenticationConverter;
import org.springframework.security.oauth2.server.resource.authentication.JwtGrantedAuthoritiesConverter;

@Bean
public JwtAuthenticationConverter jwtAuthenticationConverter() {
    JwtGrantedAuthoritiesConverter converter = new JwtGrantedAuthoritiesConverter();
    converter.setAuthoritiesClaimName("realm_access.roles");
    converter.setAuthorityPrefix("ROLE_");

    JwtAuthenticationConverter jwtConverter = new JwtAuthenticationConverter();
    jwtConverter.setJwtGrantedAuthoritiesConverter(converter);
    return jwtConverter;
}
```

Then add it in config:

```java
.oauth2ResourceServer(oauth -> oauth
    .jwt(jwt -> jwt.jwtAuthenticationConverter(jwtAuthenticationConverter()))
)
```

---

## ✅ Summary

| Step | Description                                                                   |
| ---- | ----------------------------------------------------------------------------- |
| 1.   | Spring reads `Authorization: Bearer <token>`                                  |
| 2.   | Token validated using public key from Keycloak                                |
| 3.   | Spring builds `JwtAuthenticationToken` and sets it in `SecurityContextHolder` |
| 4.   | You can access JWT claims in controllers/services                             |

---

Would you like:

* To map Keycloak **client roles or resource roles** too?
* Help integrating **Keycloak Spring Boot Starter** instead of manual setup?
