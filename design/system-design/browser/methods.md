## HTTP Methods 

### **1. GET — Retrieve Resource**

**Explanation (3–4 lines):**
GET is used to **fetch data** from the server without modifying anything.
It is **idempotent** and **safe**, meaning multiple GETs have no side effects.
Data is passed in **URL parameters**.
Commonly used for read-only operations.

**cURL Example:**

```bash
curl -X GET "https://api.example.com/users/10"
```

**Response (example):**

```json
{"id":10,"name":"John","status":"active"}
```

**Best use:** Fetching data, searching, listing.

### **2. POST — Create Resource**

**Explanation (3–4 lines):**
POST is used to **create new resources** or trigger non-idempotent actions.
It sends data in the **request body**.
Not idempotent: calling twice creates duplicates.
Often used for forms, uploads, or new entries.

**cURL Example:**

```bash
curl -X POST "https://api.example.com/users" \
  -H "Content-Type: application/json" \
  -d '{"name":"Alice","email":"alice@example.com"}'
```

**Response:**

```json
{"id":101,"name":"Alice","email":"alice@example.com"}
```

**Best use:** Creating users, orders, payments, messages.

### **3. PUT — Replace Entire Resource**

**Explanation (3–4 lines):**
PUT is **idempotent** and replaces the entire resource at the URL.
If the record doesn't exist, the server may create it.
Requires the **full object** each time.
Used for complete updates.

**cURL Example:**

```bash
curl -X PUT "https://api.example.com/users/10" \
  -H "Content-Type: application/json" \
  -d '{"id":10,"name":"John Updated","email":"new@mail.com"}'
```

**Response:**

```json
{"message":"User replaced"}
```

**Best use:** Replace full user profile, configuration objects.

### **4. PATCH — Partial Update**

**Explanation (3–4 lines):**
PATCH updates **only the changed fields**, not the whole resource.
It is **not required to be idempotent**, but usually implemented that way.
More efficient than PUT.
Suitable for partial modifications.

**cURL Example:**

```bash
curl -X PATCH "https://api.example.com/users/10" \
  -H "Content-Type: application/json" \
  -d '{"status":"inactive"}'
```

**Response:**

```json
{"id":10,"status":"inactive"}
```

**Best use:** Update only a field (status, price, name).


### **5. DELETE — Remove Resource**

**Explanation (3–4 lines):**
DELETE removes a resource at the URL.
It is **idempotent**: deleting twice has the same final state.
May return 200, 202, or 204.
Used to remove records.

**cURL Example:**

```bash
curl -X DELETE "https://api.example.com/users/10"
```

**Response:**

```json
{"message":"User deleted"}
```

**Best use:** Deleting users, files, tokens.



### **6. HEAD — Headers Only**

**Explanation (3–4 lines):**
HEAD is identical to GET but returns **only response headers**, not body.
Used to check resource availability, caching, or size.
Helps reduce bandwidth usage.
Often used by monitoring tools.

**cURL Example:**

```bash
curl -I "https://api.example.com/users/10"
```

**Response:**

```
HTTP/1.1 200 OK
Content-Type: application/json
Content-Length: 55
```

**Best use:** Health checks, pre-validation, caching checks.



### **7. OPTIONS — Supported Methods**

**Explanation (3–4 lines):**
OPTIONS returns the **allowed HTTP methods** for a URL.
Used mainly for **CORS preflight** requests.
Does not return content.
Helps clients understand API capabilities.

**cURL Example:**

```bash
curl -X OPTIONS "https://api.example.com/users" -i
```

**Response:**

```
Allow: GET, POST, PUT, PATCH, DELETE, OPTIONS
```

**Best use:** API discovery, CORS, client compatibility checks.



### **8. TRACE — Diagnostic Loopback**

**Explanation (3–4 lines):**
TRACE returns the received request exactly as the server sees it.
Used for debugging proxies or connection issues.
Often disabled for security reasons.
Can reveal sensitive headers.

**cURL Example (if enabled):**

```bash
curl -X TRACE "https://api.example.com/debug"
```

**Response:**

```
TRACE /debug HTTP/1.1
Host: api.example.com
```

**Best use:** Debugging HTTP intermediaries (rare).



More

✅ **Full table comparing all HTTP methods**
✅ **REST best-practices mapping (CRUD → methods)**
✅ **Mermaid sequence diagram of an HTTP request/response lifecycle**

