Here's a complete setup to use **NGINX** as an **API Gateway** for an `employee-service` running locally (e.g., on `localhost:8081`).

---

## ✅ Goal

* Use **NGINX** as a lightweight API Gateway.
* Route all `/employees/**` requests to `employee-service`.
* Add logging and security headers.
* Run everything using **Docker**.

---

## 📦 Project Structure

```
nginx-api-gateway/
├── nginx.conf
└── docker-compose.yml
```

---

## 🔧 Step 1: `nginx.conf` – Gateway Configuration

```nginx
worker_processes 1;

events { worker_connections 1024; }

http {
    include       mime.types;
    default_type  application/octet-stream;

    access_log /var/log/nginx/access.log;
    error_log  /var/log/nginx/error.log;

    sendfile        on;
    keepalive_timeout  65;

    server {
        listen 80;

        # Route /employees to employee-service
        location /employees/ {
            proxy_pass         http://employee-service:8081/;
            proxy_set_header   Host $host;
            proxy_set_header   X-Real-IP $remote_addr;
            proxy_set_header   X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header   X-Forwarded-Proto $scheme;
            add_header         X-Gateway-Name "NGINX-Gateway";
        }

        # Optional: Health check endpoint
        location /health {
            return 200 "OK\n";
        }
    }
}
```

---

## 🐳 Step 2: `docker-compose.yml`

```yaml
version: "3.8"

services:
  nginx-gateway:
    image: nginx:1.25
    container_name: nginx-gateway
    ports:
      - "8080:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - employee-service

  employee-service:
    image: your-employee-service-image
    container_name: employee-service
    build:
      context: ../employee-service  # adjust path if needed
    ports:
      - "8081:8081"
```

> 🔁 Replace `your-employee-service-image` with your actual Docker image or a build context.

---

## 🧪 Testing

Assume your employee-service has:

```http
GET http://localhost:8081/employees
```

Then access it via NGINX Gateway:

```http
GET http://localhost:8080/employees
```

You should see the same result, with an additional header:

```
X-Gateway-Name: NGINX-Gateway
```

---

## 🛡️ Optional Enhancements

### 🔐 Add Basic Authentication:

In `nginx.conf` under `location`:

```nginx
auth_basic "Restricted";
auth_basic_user_file /etc/nginx/.htpasswd;
```

Then mount the `.htpasswd` file using volumes.

---

### 🚫 Restrict Access by IP:

```nginx
allow 192.168.0.0/16;
deny all;
```

---

### 🔁 Rewrite URL Paths:

```nginx
location /api/v1/employees/ {
    rewrite ^/api/v1/employees/(.*)$ /employees/$1 break;
    proxy_pass http://employee-service:8081/;
}
```

---

## ✅ Summary

| Feature         | Supported                      |
| --------------- | ------------------------------ |
| Routing         | ✅ Yes                          |
| Path rewrites   | ✅ Yes                          |
| Load balancing  | ✅ Yes (static upstream or DNS) |
| Auth headers    | ✅ Yes                          |
| TLS termination | ✅ Yes (with certs)             |
| Rate limiting   | ✅ Yes (with modules)           |
| Docker-friendly | ✅ Yes                          |

---

Would you like me to extend this to include **rate limiting**, **JWT token validation**, or **HTTPS with certs**?


Great! Let's extend the **NGINX API Gateway** setup to:

---

## 🎯 **Use Keycloak to validate JWT tokens** for requests routed to the `employee-service`.

We'll use **NGINX with the `lua-resty-openidc` module** for Keycloak JWT validation.

---

## 🧱 Architecture Overview

```
Client → NGINX Gateway (JWT Validation via Keycloak) → Employee Microservice
```

---

## ✅ Prerequisites

* Keycloak is up and running at e.g., `http://localhost:8080`
* You have a **realm** (e.g., `myrealm`) and **client** (e.g., `gateway-client`)
* JWT is issued by Keycloak and passed in `Authorization: Bearer <token>` header

---

## 📦 File Structure

```
nginx-api-gateway/
├── nginx.conf
├── docker-compose.yml
├── keycloak/
│   └── ... (optional docker-compose setup)
└── lua/
    └── jwt-auth.lua
```

---

## 🐳 Step 1: `docker-compose.yml`

Use **OpenResty** (NGINX + Lua support) for JWT verification.

```yaml
version: "3.8"

services:
  nginx-gateway:
    image: openresty/openresty:1.21.4.1-3-alpine
    container_name: nginx-gateway
    ports:
      - "8080:80"
    volumes:
      - ./nginx.conf:/usr/local/openresty/nginx/conf/nginx.conf:ro
      - ./lua:/etc/nginx/lua
    depends_on:
      - employee-service

  employee-service:
    image: your-employee-service-image
    ports:
      - "8081:8081"
```

---

## 🔒 Step 2: `lua/jwt-auth.lua` — JWT Verification using OpenID Connect

```lua
local opts = {
  discovery = "http://host.docker.internal:8080/realms/myrealm/.well-known/openid-configuration",
  client_id = "gateway-client",
  client_secret = "client-secret-if-confidential",
  redirect_uri_path = "/redirect_uri",
  token_signing_alg_values_supported = { "RS256" }
}

local res, err = require("resty.openidc").bearer_jwt_verify(opts)

if err then
  ngx.status = 401
  ngx.say("Unauthorized: ", err)
  ngx.exit(ngx.HTTP_UNAUTHORIZED)
end

-- Token is valid, set user info to header (optional)
ngx.req.set_header("X-User", res.payload.preferred_username)
```

---

## ⚙️ Step 3: `nginx.conf`

```nginx
worker_processes 1;
events { worker_connections 1024; }

http {
  lua_package_path "/etc/nginx/lua/?.lua;;";

  server {
    listen 80;

    location /employees/ {
      access_by_lua_file /etc/nginx/lua/jwt-auth.lua;

      proxy_pass         http://employee-service:8081/;
      proxy_set_header   Host $host;
      proxy_set_header   X-Real-IP $remote_addr;
      proxy_set_header   X-Forwarded-For $proxy_add_x_forwarded_for;
      proxy_set_header   X-Forwarded-Proto $scheme;
    }

    location /health {
      return 200 "OK";
    }
  }
}
```

---

## ✅ Test Flow

1. Get token from Keycloak:

   ```bash
   curl -X POST http://localhost:8080/realms/myrealm/protocol/openid-connect/token \
     -d "grant_type=password&client_id=gateway-client&username=admin&password=admin"
   ```

2. Access the service through NGINX:

   ```bash
   curl http://localhost:8080/employees \
     -H "Authorization: Bearer <your_jwt_token>"
   ```

If the token is valid, the request reaches `employee-service`. If not, it returns 401.

---

## 🧠 Summary

| Feature                 | NGINX (OpenResty + Lua)   |
| ----------------------- | ------------------------- |
| JWT validation          | ✅ Via `lua-resty-openidc` |
| OAuth2 via Keycloak     | ✅ Supported               |
| Lightweight             | ✅ Yes                     |
| Secure                  | ✅ Yes                     |
| Run locally with Docker | ✅ Yes                     |

---

Would you like me to also:

* ✅ Add rate limiting?
* ✅ Validate roles (e.g., only allow role `admin`) from the JWT?
* ✅ Set up a Keycloak container for testing?


Perfect! Here's a **complete working setup** that uses:

* **NGINX (OpenResty)** as API Gateway
* **Keycloak** for issuing JWT tokens
* **Employee Microservice** (can be a mock or your real one)
* **JWT token validation with role-based access** in Lua
* **Optional Rate Limiting**

---

## ✅ Architecture Overview

```
Client
  ↓ (JWT)
NGINX Gateway (OpenResty + Lua)
  ↓ (Forward if JWT is valid & has role)
Employee Microservice
```

---

## 🏗️ Final File Structure

```
nginx-api-gateway/
├── docker-compose.yml
├── nginx.conf
├── lua/
│   └── jwt-auth.lua
└── keycloak/
    └── (auto setup Keycloak with realm + user + client)
```

---

### 🐳 `docker-compose.yml`

```yaml
version: "3.9"

services:
  nginx-gateway:
    image: openresty/openresty:1.21.4.1-3-alpine
    container_name: nginx-gateway
    ports:
      - "8080:80"
    volumes:
      - ./nginx.conf:/usr/local/openresty/nginx/conf/nginx.conf:ro
      - ./lua:/etc/nginx/lua
    depends_on:
      - keycloak
      - employee-service

  employee-service:
    image: kennethreitz/httpbin
    container_name: employee-service
    ports:
      - "8081:80" # httpbin uses port 80 internally

  keycloak:
    image: quay.io/keycloak/keycloak:24.0.1
    container_name: keycloak
    command: start-dev --import-realm
    ports:
      - "8085:8080"
    environment:
      KEYCLOAK_ADMIN: admin
      KEYCLOAK_ADMIN_PASSWORD: admin
    volumes:
      - ./keycloak:/opt/keycloak/data/import
```

---

### 📂 `keycloak/realm-export.json` (auto-setup realm, user, client)

```json
{
  "realm": "myrealm",
  "enabled": true,
  "clients": [
    {
      "clientId": "gateway-client",
      "publicClient": true,
      "directAccessGrantsEnabled": true,
      "redirectUris": ["*"]
    }
  ],
  "users": [
    {
      "username": "user1",
      "enabled": true,
      "credentials": [
        { "type": "password", "value": "user1" }
      ],
      "realmRoles": ["employee"]
    }
  ],
  "roles": {
    "realm": {
      "employee": {}
    }
  }
}
```

---

### 🔐 `lua/jwt-auth.lua` (JWT validation + role check)

```lua
local openidc = require("resty.openidc")

local opts = {
  discovery = "http://keycloak:8080/realms/myrealm/.well-known/openid-configuration",
  client_id = "gateway-client",
  token_signing_alg_values_supported = { "RS256" }
}

local res, err = openidc.bearer_jwt_verify(opts)

if err then
  ngx.status = 401
  ngx.say("Unauthorized: Invalid token - ", err)
  return ngx.exit(ngx.HTTP_UNAUTHORIZED)
end

local roles = res.payload.realm_access and res.payload.realm_access.roles or {}
local hasRole = false
for _, r in ipairs(roles) do
  if r == "employee" then
    hasRole = true
    break
  end
end

if not hasRole then
  ngx.status = 403
  ngx.say("Forbidden: role 'employee' required")
  return ngx.exit(ngx.HTTP_FORBIDDEN)
end

-- Optional: forward user info
ngx.req.set_header("X-User", res.payload.preferred_username)
```

---

### ⚙️ `nginx.conf`

```nginx
worker_processes 1;
events { worker_connections 1024; }

http {
  lua_package_path "/etc/nginx/lua/?.lua;;";

  server {
    listen 80;

    location /employees/ {
      access_by_lua_file /etc/nginx/lua/jwt-auth.lua;

      proxy_pass         http://employee-service:80/;
      proxy_set_header   Host $host;
      proxy_set_header   X-Real-IP $remote_addr;
      proxy_set_header   X-Forwarded-For $proxy_add_x_forwarded_for;
      proxy_set_header   X-Forwarded-Proto $scheme;
    }

    location /health {
      return 200 "OK";
    }
  }
}
```

---

## 🚀 How to Run

```bash
docker-compose up --build
```

---

## 🧪 How to Test It

### 🔐 Step 1: Get Token

```bash
curl -X POST http://localhost:8085/realms/myrealm/protocol/openid-connect/token \
  -d "client_id=gateway-client" \
  -d "grant_type=password" \
  -d "username=user1" \
  -d "password=user1" \
  -H "Content-Type: application/x-www-form-urlencoded"
```

Copy the `access_token`.

---

### 📡 Step 2: Call API Gateway

```bash
curl http://localhost:8080/employees/get \
  -H "Authorization: Bearer <access_token>"
```

Should return a 200 response from httpbin (acting as employee-service).

---

## ✅ Summary

| Feature                                 | Included                     |
| --------------------------------------- | ---------------------------- |
| NGINX Gateway                           | ✅                            |
| JWT Token Validation via Keycloak       | ✅                            |
| Role-based Access                       | ✅ (requires "employee" role) |
| Docker-ized Setup                       | ✅                            |
| Secure Gateway                          | ✅                            |
| Ready-to-run Keycloak with realm + user | ✅                            |

---

Would you like me to extend this with:

* ✅ Rate limiting via NGINX?
* ✅ HTTPS support with self-signed certs?
* ✅ Keycloak login via browser redirect (OIDC full flow)?

Let me know!


Great! Let's extend the setup in **3 directions**:

---

## ✅ 1. Add Rate Limiting in NGINX

### 🔧 Update `nginx.conf`

Add this inside the `http {}` block:

```nginx
# Rate limiting zone: 5 requests per second with a burst of 10
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=5r/s;
```

Update the `/employees/` location block:

```nginx
location /employees/ {
    limit_req zone=api_limit burst=10 nodelay;

    access_by_lua_file /etc/nginx/lua/jwt-auth.lua;

    proxy_pass http://employee-service:80/;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
}
```

---

## ✅ 2. Enable HTTPS with Self-Signed Cert

### 🔧 Step 1: Generate certs (on host machine)

```bash
mkdir certs
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout certs/nginx.key -out certs/nginx.crt \
  -subj "/C=IN/ST=Karnataka/L=Bangalore/O=Example/OU=IT/CN=localhost"
```

### 🐳 Step 2: Update `docker-compose.yml`

Mount the certs:

```yaml
    volumes:
      - ./nginx.conf:/usr/local/openresty/nginx/conf/nginx.conf:ro
      - ./lua:/etc/nginx/lua
      - ./certs:/etc/nginx/certs:ro
```

---

### 🔧 Step 3: Update `nginx.conf` to add HTTPS

```nginx
server {
    listen 443 ssl;

    ssl_certificate /etc/nginx/certs/nginx.crt;
    ssl_certificate_key /etc/nginx/certs/nginx.key;

    location /employees/ {
        limit_req zone=api_limit burst=10 nodelay;

        access_by_lua_file /etc/nginx/lua/jwt-auth.lua;

        proxy_pass http://employee-service:80/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /health {
        return 200 "OK";
    }
}

# Optional: redirect HTTP to HTTPS
server {
    listen 80;
    return 301 https://$host$request_uri;
}
```

---

## ✅ 3. Add OIDC Redirect Login Flow (Browser Login)

If you want full browser login (OIDC Code Flow):

1. Change Keycloak client to a **confidential** client with redirect URIs like:

   ```
   http://localhost:8080/callback
   ```

2. Create a new route in NGINX:

```nginx
location /auth {
    content_by_lua_block {
        local openidc = require("resty.openidc")
        local opts = {
            discovery = "http://keycloak:8080/realms/myrealm/.well-known/openid-configuration",
            client_id = "gateway-client",
            client_secret = "client-secret",
            redirect_uri = "https://localhost/auth/callback",
            scope = "openid email profile"
        }

        local res, err = openidc.authenticate(opts)

        if err then
            ngx.status = 500
            ngx.say("OIDC auth failed: ", err)
            return ngx.exit(ngx.HTTP_INTERNAL_SERVER_ERROR)
        end

        ngx.say("Hello, ", res.user.preferred_username)
    }
}
```

Now visiting `https://localhost/auth` will redirect to Keycloak login.

---

## 🧪 Test Endpoints

| Endpoint                      | Description                     |
| ----------------------------- | ------------------------------- |
| `https://localhost/employees` | Secured with JWT & rate-limit   |
| `https://localhost/health`    | Open health check               |
| `https://localhost/auth`      | Full browser login via Keycloak |

---

## 🧠 Summary

| Feature                       | Status  |
| ----------------------------- | ------- |
| ✅ JWT auth via Keycloak       | Done    |
| ✅ Role check in Lua           | Done    |
| ✅ Dockerized Keycloak         | Done    |
| ✅ Rate limiting               | ✅ Added |
| ✅ HTTPS with self-signed cert | ✅ Added |
| ✅ OIDC browser login          | ✅ Added |

---

Would you like:

* 🔄 A fully zipped version of this repo?
* 🧪 Postman collection for testing?
* 🧰 Extend it with Redis session storage or cache for JWTs?

Let me know!
