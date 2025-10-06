Excellent — let’s go through **all HTTP response status codes**, grouped by category (1xx–5xx), with clear explanations, usage examples, and important details for developers and system designers.

---

## 🌐 HTTP Status Codes Overview

HTTP response codes are **standardized numeric responses** returned by a server to indicate the **result** of a client’s request.
They are grouped into five classes:

| Code Range | Category      | Meaning                                          |
| ---------- | ------------- | ------------------------------------------------ |
| **1xx**    | Informational | Request received, continuing process             |
| **2xx**    | Success       | Request successfully received and processed      |
| **3xx**    | Redirection   | Further action required (redirects, cache, etc.) |
| **4xx**    | Client Error  | Request error due to client                      |
| **5xx**    | Server Error  | Request failed due to server                     |

---

## 🟦 1xx — Informational Responses

| Code    | Name                | Description                                             | Example Use                                           |
| ------- | ------------------- | ------------------------------------------------------- | ----------------------------------------------------- |
| **100** | Continue            | Client can continue sending request body.               | Sent after headers when using `Expect: 100-continue`. |
| **101** | Switching Protocols | Server is switching protocols (e.g., to WebSocket).     | HTTP → WebSocket upgrade handshake.                   |
| **102** | Processing          | Server received request but not yet completed (WebDAV). | Long-running operations like `COPY` or `MOVE`.        |
| **103** | Early Hints         | Server sends preliminary headers before final response. | Used to preload linked resources.                     |

---

## 🟩 2xx — Success Responses

| Code    | Name                          | Description                                                | Example Use                                   |
| ------- | ----------------------------- | ---------------------------------------------------------- | --------------------------------------------- |
| **200** | OK                            | Request succeeded, response body contains result.          | Standard response for successful GET or POST. |
| **201** | Created                       | Resource created successfully.                             | POST `/users` created a new user.             |
| **202** | Accepted                      | Request accepted for processing but not completed.         | Async jobs or background task queues.         |
| **203** | Non-Authoritative Information | Response from a transforming proxy, not the origin server. | Proxy-modified metadata.                      |
| **204** | No Content                    | Request succeeded, no content returned.                    | DELETE `/users/123` success.                  |
| **205** | Reset Content                 | Client should reset document view (rare).                  | Form reset after submission.                  |
| **206** | Partial Content               | Partial data delivered (used with Range header).           | Video streaming or resume download.           |
| **207** | Multi-Status                  | Multiple independent results (WebDAV).                     | Batch operations on multiple resources.       |
| **208** | Already Reported              | Repeated resource already reported (WebDAV).               | Complex multi-resource responses.             |
| **226** | IM Used                       | Response includes result of instance manipulations.        | HTTP delta encoding (rare).                   |

---

## 🟨 3xx — Redirection Responses

| Code    | Name                     | Description                                | Example Use                           |
| ------- | ------------------------ | ------------------------------------------ | ------------------------------------- |
| **300** | Multiple Choices         | Multiple options for resource (rare).      | Different formats or languages.       |
| **301** | Moved Permanently        | Resource moved, use new URL permanently.   | HTTP → HTTPS migration.               |
| **302** | Found                    | Resource temporarily moved to another URL. | Legacy redirect, replaced by 303/307. |
| **303** | See Other                | Redirect to another URI with GET method.   | POST `/upload` → GET `/status`.       |
| **304** | Not Modified             | Cached resource still valid.               | Used with `If-Modified-Since`.        |
| **305** | Use Proxy *(deprecated)* | Resource must be accessed via a proxy.     | Deprecated for security.              |
| **306** | Switch Proxy *(unused)*  | Reserved, not used anymore.                | Historical.                           |
| **307** | Temporary Redirect       | Redirect, preserving original method.      | POST redirected to another endpoint.  |
| **308** | Permanent Redirect       | Like 301, but preserves method.            | Permanent API endpoint move.          |

---

## 🟥 4xx — Client Error Responses

| Code    | Name                            | Description                                           | Example Use                              |
| ------- | ------------------------------- | ----------------------------------------------------- | ---------------------------------------- |
| **400** | Bad Request                     | Malformed syntax or invalid data.                     | Missing required JSON field.             |
| **401** | Unauthorized                    | Authentication required or failed.                    | Missing/invalid JWT token.               |
| **402** | Payment Required                | Reserved for future use (used in APIs for billing).   | Paywall APIs.                            |
| **403** | Forbidden                       | Authenticated, but not allowed.                       | User lacks permission for resource.      |
| **404** | Not Found                       | Resource not found.                                   | Non-existent URL or ID.                  |
| **405** | Method Not Allowed              | HTTP method not supported.                            | PUT not allowed on `/login`.             |
| **406** | Not Acceptable                  | Server cannot produce acceptable content type.        | `Accept: application/xml` not supported. |
| **407** | Proxy Authentication Required   | Client must authenticate with proxy.                  | Corporate proxies.                       |
| **408** | Request Timeout                 | Client took too long to send request.                 | Idle connections closed.                 |
| **409** | Conflict                        | Request conflicts with current resource state.        | Version conflict or duplicate record.    |
| **410** | Gone                            | Resource permanently deleted.                         | Legacy API endpoint removed.             |
| **411** | Length Required                 | Missing Content-Length header.                        | Needed for non-chunked POST.             |
| **412** | Precondition Failed             | `If-Match` or `If-Unmodified-Since` condition failed. | Used for concurrency control.            |
| **413** | Payload Too Large               | Request entity too large.                             | File upload exceeds limit.               |
| **414** | URI Too Long                    | URI exceeds server limit.                             | Excessive query string data.             |
| **415** | Unsupported Media Type          | Unsupported Content-Type.                             | Sending XML where JSON expected.         |
| **416** | Range Not Satisfiable           | Invalid range header value.                           | Video byte range beyond end.             |
| **417** | Expectation Failed              | `Expect` header requirement failed.                   | `Expect: 100-continue` rejected.         |
| **418** | I'm a teapot ☕                  | Joke status from RFC 2324.                            | Easter egg only.                         |
| **421** | Misdirected Request             | Request sent to wrong server.                         | HTTP/2 connection reuse issue.           |
| **422** | Unprocessable Entity            | Request understood but invalid semantics.             | Validation errors in REST APIs.          |
| **423** | Locked                          | Resource is locked.                                   | WebDAV or document editing.              |
| **424** | Failed Dependency               | Dependent request failed.                             | Chained WebDAV operations.               |
| **425** | Too Early                       | Request replayed too soon (HTTP/2).                   | Retry-attack prevention.                 |
| **426** | Upgrade Required                | Client must upgrade protocol (e.g., to TLS).          | `Upgrade: TLS/1.2`.                      |
| **428** | Precondition Required           | Server requires conditional request.                  | ETag-based updates.                      |
| **429** | Too Many Requests               | Rate limit exceeded.                                  | API throttling.                          |
| **431** | Request Header Fields Too Large | Header section too big.                               | Excessive cookies or headers.            |
| **451** | Unavailable For Legal Reasons   | Censorship or legal restriction.                      | GDPR or DMCA takedown.                   |

---

## 🟥 5xx — Server Error Responses

| Code    | Name                            | Description                                 | Example Use                   |
| ------- | ------------------------------- | ------------------------------------------- | ----------------------------- |
| **500** | Internal Server Error           | Generic server-side failure.                | Uncaught exception.           |
| **501** | Not Implemented                 | Server does not support the request method. | Unsupported HTTP verb.        |
| **502** | Bad Gateway                     | Invalid response from upstream server.      | Reverse proxy failure.        |
| **503** | Service Unavailable             | Server temporarily overloaded or down.      | Maintenance mode.             |
| **504** | Gateway Timeout                 | Upstream server didn’t respond in time.     | Proxy timeout.                |
| **505** | HTTP Version Not Supported      | Unsupported HTTP version.                   | Client uses HTTP/0.9.         |
| **506** | Variant Also Negotiates         | Negotiation configuration error.            | Content negotiation loop.     |
| **507** | Insufficient Storage            | Server can’t store representation.          | WebDAV or full disk.          |
| **508** | Loop Detected                   | Infinite loop detected in processing.       | Recursive dependency.         |
| **510** | Not Extended                    | Further extensions required.                | Custom extension negotiation. |
| **511** | Network Authentication Required | Client must authenticate to network.        | Captive portal Wi-Fi login.   |

---

## 🧠 Common Usage Scenarios (Real-World)

| Scenario                | Request                       | Typical Response            |
| ----------------------- | ----------------------------- | --------------------------- |
| Fetching resource       | GET `/users/1`                | **200 OK**                  |
| Creating new resource   | POST `/users`                 | **201 Created**             |
| Asynchronous processing | POST `/jobs`                  | **202 Accepted**            |
| Resource deleted        | DELETE `/users/2`             | **204 No Content**          |
| Client validation error | POST invalid JSON             | **400 Bad Request**         |
| Unauthorized            | GET `/admin`                  | **401 Unauthorized**        |
| Forbidden access        | GET `/admin` (non-admin user) | **403 Forbidden**           |
| Missing resource        | GET `/users/999`              | **404 Not Found**           |
| API rate limit hit      | Multiple rapid requests       | **429 Too Many Requests**   |
| Server down             | Any request                   | **503 Service Unavailable** |

---

## 🧩 Quick Summary Table

| Category | Range   | Meaning       |
| -------- | ------- | ------------- |
| 1xx      | 100–199 | Informational |
| 2xx      | 200–299 | Success       |
| 3xx      | 300–399 | Redirection   |
| 4xx      | 400–499 | Client Error  |
| 5xx      | 500–599 | Server Error  |

---

Would you like me to extend this with **visual mapping (mermaid diagram)** showing how different **status code flows** are used in a **REST API lifecycle** (e.g., Create → Conflict → Success → Redirect)?
