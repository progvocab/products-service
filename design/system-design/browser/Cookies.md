# Cookies

Cookies are small pieces of data stored in the user’s browser by a website. They are commonly classified **by lifespan, scope, purpose, and security attributes**. Below is a **complete, structured breakdown** that’s useful both conceptually and for interviews.



> *“Cookies can be classified by lifespan (session, persistent), ownership (first-party, third-party), purpose (essential, analytics, advertising), and security attributes like Secure, HttpOnly, and SameSite.”*

 

##   Based on Lifespan (Most Common Classification)

### 1. Session Cookies

* Stored **in memory**, not on disk.
* Deleted when the **browser is closed**.
* Used to maintain session state (login, shopping cart).

**Example:**
Session ID after login.

 

### 2. Persistent Cookies

* Stored on disk with an **expiration date**.
* Remain even after browser restart until expiry.
* Used for “Remember me”, preferences.

**Example:**
Language selection cookie valid for 30 days.

 

##   Based on Domain / Scope

### 3. First-Party Cookies

* Set by the **same domain** the user is visiting.
* Accessible only by that domain.
* Generally considered safer.

**Example:**
`example.com` setting login cookie.

 

### 4. Third-Party Cookies

* Set by a **different domain** (ads, analytics).
* Commonly blocked by modern browsers.
* Used for cross-site tracking.

**Example:**
`ads.google.com` cookie on `example.com`.

 

##   Based on Purpose / Function

### 5. Essential (Strictly Necessary) Cookies

* Required for website to function.
* No consent usually required (GDPR).
* Used for:

  * Authentication
  * Load balancing
  * CSRF protection

 

### 6. Functional Cookies

* Store user preferences.
* Improve user experience.

**Example:**
Theme (dark/light), language.

 

### 7. Performance / Analytics Cookies

* Collect anonymous usage statistics.
* Help improve website performance.

**Example:**
Google Analytics cookies.

 
### 8. Advertising / Targeting Cookies

* Track browsing behavior.
* Used for personalized ads.
* Highly regulated.

 
##   Based on Security Attributes (Very Important)

### 9. Secure Cookies

* Sent **only over HTTPS**.
* Prevent interception over plaintext HTTP.

```http
Set-Cookie: sessionId=abc; Secure
```

 

### 10. HttpOnly Cookies

* Not accessible via JavaScript.
* Protect against **XSS attacks**.

```http
Set-Cookie: sessionId=abc; HttpOnly
```

 

### 11. SameSite Cookies

Controls **cross-site cookie sending**.

| Value  | Behavior                     |
| ------ | ---------------------------- |
| Strict | Never sent cross-site        |
| Lax    | Sent on top-level navigation |
| None   | Always sent (must be Secure) |

```http
Set-Cookie: sessionId=abc; SameSite=Lax
```
 ### Cookie Security Attributes  

> Cookie attributes control **who can read a cookie, when it is sent, and how it is transmitted**, directly impacting XSS, CSRF, session hijacking, and privacy.

*“Cookie security attributes like Secure, HttpOnly, SameSite, and the __Host- prefix collectively protect against MITM, XSS, CSRF, and cookie injection attacks.”*
 

### `Secure`

 

* Cookie is sent **only over HTTPS** connections.
* Prevents cookie leakage via plaintext HTTP.

 

* Protects against **Man-in-the-Middle (MITM)** attacks.
* Essential for authentication cookies.

 

```http
Set-Cookie: sessionId=abc123; Secure
```

 

-  Always use for auth/session cookies
-  Avoid using cookies without HTTPS in production

 

### `HttpOnly`

 

* Cookie is **not accessible via JavaScript** (`document.cookie`).
* Only sent via HTTP headers.

 

* Protects against **Cross-Site Scripting (XSS)**.
* Even if JS is compromised, cookie cannot be stolen.

 

```http
Set-Cookie: sessionId=abc123; HttpOnly
```

 

- Mandatory for session cookies
- Do not use for cookies that must be read by JS (e.g. UI preferences)

 

### `SameSite`

 

* Controls **whether cookies are sent in cross-site requests**.
* Primary defense against **CSRF attacks**.

 

| Value    | Behavior                           |
| -------- | ---------------------------------- |
| `Strict` | Never sent cross-site              |
| `Lax`    | Sent on top-level navigation (GET) |
| `None`   | Always sent (requires `Secure`)    |

 

```http
Set-Cookie: sessionId=abc123; SameSite=Lax
```

 

* `Strict`: Highly sensitive apps (banking)
* `Lax`: Most web apps (default & recommended)
* `None`: Required for cross-site iframes, SSO

 

### `Domain`

 

* Specifies **which domains can receive the cookie**.
* Can allow subdomains.

 

```http
Set-Cookie: sessionId=abc; Domain=example.com
```

 

* Broader domain = **larger attack surface**
* Avoid unnecessary subdomain sharing

 

- Use the **narrowest domain possible**

 

### `Path`

 

* Restricts cookie to a specific **URL path**.

 

```http
Set-Cookie: sessionId=abc; Path=/app
```

 

* Limits exposure to unrelated endpoints.
* Prevents accidental cookie leaks.

 

- Restrict path when possible

 

### `Expires`

 

* Sets a **fixed expiration date/time**.
* Makes cookie persistent.

 

```http
Set-Cookie: token=abc; Expires=Wed, 21 Oct 2026 07:28:00 GMT
```

 

* Longer lifetime = higher risk if stolen.
* Session cookies disappear on browser close.

 

- Keep lifetimes short for auth cookies

 

### `Max-Age`

 

* Cookie expires after **N seconds**.
* Preferred over `Expires`.

 
```http
Set-Cookie: token=abc; Max-Age=3600
```

 
* Precise control over token lifetime.

 
- Use `Max-Age` instead of `Expires`

 
### `__Secure-` Prefix

 
* Enforces:

  * Must be set over HTTPS
  * Must include `Secure`

 
```http
Set-Cookie: __Secure-sessionId=abc; Secure
```

 
* Prevents cookies from being set by insecure origins.

 

### `__Host-` Prefix (Most Secure)

 
* Enforces:

  * `Secure`
  * **No `Domain` attribute**
  * `Path=/`
  * HTTPS only

 
```http
Set-Cookie: __Host-sessionId=abc; Secure; Path=/
```

 
* Prevents subdomain cookie injection.
* Strongest cookie isolation.

 
-  Use for session cookies when possible
 
### Priority (Less Known but Important)
 

* Defines eviction priority when browser storage is full.
 

* `Low`
* `Medium`
* `High`
 

```http
Set-Cookie: sessionId=abc; Priority=High
```

 

### Partitioned (CHIPS – Modern Browsers)

 
* Cookie is **partitioned per top-level site**.
* Prevents cross-site tracking.
 

```http
Set-Cookie: sessionId=abc; Secure; SameSite=None; Partitioned
```

 

* Third-party embeds that still need state.

 

### `SameParty` (Experimental / Limited Support)

 

* Allows cookie sharing within a **set of related domains**.

  Limited browser support; use cautiously.
 

 

| Attribute         | Protects Against    |
| ----------------- | ------------------- |
| Secure            | MITM                |
| HttpOnly          | XSS                 |
| SameSite          | CSRF                |
| Domain            | Subdomain abuse     |
| Path              | Over-exposure       |
| Expires / Max-Age | Token theft risk    |
| __Host-           | Cookie injection    |
| __Secure-         | Insecure origins    |
| Partitioned       | Cross-site tracking |

 
  Production-Grade Secure Cookie Example

```http
Set-Cookie: __Host-sessionId=abc123;
  Secure;
  HttpOnly;
  SameSite=Lax;
  Path=/;
  Max-Age=1800
```

 

 

 

###   Based on Storage Location

### 12. In-Memory Cookies

* Exist only during session.
* Faster, but volatile.



### 13. Disk Cookies

* Persisted until expiration.
* Used for long-term preferences.



###   Special / Less Common Types

### 14. Zombie Cookies

* Recreated after deletion using multiple storage mechanisms.
* Considered **privacy-invasive**.

 

### 15. Supercookies

* Stored outside standard cookie storage.
* Hard to detect/delete.
* Often used by ISPs or malicious actors.

 

### 16. Flash Cookies (LSOs)

* Stored by Adobe Flash (mostly obsolete).
* Survive browser cookie deletion.

 

| Type        | Key Property       | Use Case           |
| ----------- | ------------------ | ------------------ |
| Session     | Browser session    | Login session      |
| Persistent  | Expiration date    | Remember user      |
| First-party | Same domain        | Core functionality |
| Third-party | Different domain   | Ads, tracking      |
| Secure      | HTTPS only         | Prevent sniffing   |
| HttpOnly    | JS blocked         | XSS protection     |
| SameSite    | Cross-site control | CSRF protection    |
| Essential   | Mandatory          | Site operation     |
| Analytics   | Metrics            | Improve UX         |
| Advertising | Targeted ads       | Marketing          |

 

 ## Cookie Injection

 **Cookie injection** is a **web security attack** where an attacker **forces a browser to store or overwrite a cookie** for a target website, potentially **manipulating session state, authentication, or application behavior**.
> *“Cookie injection is an attack where an attacker forces a browser to store a malicious cookie for a target domain, often via subdomains, header injection, or insecure transport, leading to session fixation or privilege escalation.”*

Think of it as:

> *“The attacker doesn’t steal your cookie — they trick your browser into accepting a malicious one.”*

 

### What Exactly Happens?

An attacker causes the victim’s browser to receive a **`Set-Cookie` header** that:

* Belongs to a **legitimate domain**
* Contains **attacker-controlled values**
* Is stored by the browser and later sent to the real site

If the application **trusts cookie values**, the attacker may:

* Hijack sessions
* Bypass authentication
* Elevate privileges
* Poison application logic
 
 

### Vulnerable scenario

* App trusts a cookie called `role=admin`
* App does **not validate** the value server-side

### Attack

Attacker injects:

```http
Set-Cookie: role=admin; Domain=example.com; Path=/
```

Now every request from the victim:

```http
Cookie: role=admin
```

Result   **Privilege escalation**

 

### Common Ways Cookie Injection Happens

###   Subdomain Cookie Injection (Most Common)

If an attacker controls:

```
evil.example.com
```

They can set:

```http
Set-Cookie: sessionId=attacker; Domain=example.com
```

Browser sends this cookie to:

```
www.example.com
```

  **Root cause**:
Cookies with `Domain=example.com` are valid for **all subdomains**.

 

###   Response Splitting / Header Injection

If an application reflects user input into HTTP headers:

```http
X-User: <user-input>
```

Attacker input:

```
\r\nSet-Cookie: sessionId=evil
```

Result   Browser stores injected cookie.

 

###   Insecure HTTP + MITM

* Cookies set **without Secure**
* Attacker intercepts traffic and injects cookies.

 

###   XSS-Assisted Cookie Injection

* Malicious JavaScript sets cookies via:

```js
document.cookie = "sessionId=evil"
```

If cookie is not `HttpOnly`, it can be overwritten.
 

###   Why Cookie Injection Is Dangerous

| Impact               | Explanation                                      |
| -------------------- | ------------------------------------------------ |
| Session fixation     | Victim logs in using attacker-controlled session |
| Privilege escalation | Fake role/flag cookies                           |
| Account takeover     | Combined with login                              |
| CSRF amplification   | Malicious state cookies                          |
| Logic abuse          | App trusts cookie flags                          |

 

###   How to Prevent Cookie Injection

###   Use `__Host-` Prefix (Best Defense)

```http
Set-Cookie: __Host-sessionId=abc;
  Secure;
  Path=/
```

✔ Prevents:

* Domain attribute
* Subdomain injection
* Insecure origins
 

###   Never Trust Client Cookies

* Treat cookies as **untrusted input**
* Validate session data server-side (DB / cache)

 

###   Use `HttpOnly`

* Prevent JS overwriting cookies

 

###   Set Narrow `Domain` and `Path`

* Avoid `Domain=example.com` unless required

 

###   Enforce HTTPS + HSTS

* Prevent MITM injection

 

###   Sanitize Headers

* Prevent CRLF injection
* Validate all header inputs

 

###   Rotate Session IDs on Login

* Prevent session fixation attacks

Secure Cookie Example

```http
Set-Cookie: __Host-sessionId=xyz;
  Secure;
  HttpOnly;
  SameSite=Lax;
  Path=/
```

 


 

### Cookie Injection vs Cookie Theft

| Attack           | Goal                    |
| ---------------- | ----------------------- |
| Cookie theft     | Steal victim’s cookie   |
| Cookie injection | Plant attacker’s cookie |

 
 
 

More:

* Compare **cookies vs localStorage vs sessionStorage**
* Explain **cookies in authentication (JWT vs session ID)**
* Dive into **CSRF and SameSite cookie behavior**
* Map cookies to **real production security setups**

* **JWT in cookies vs Authorization header**
* **CSRF attack flow with SameSite examples**
* **Browser behavior differences (Chrome vs Safari vs Firefox)**
* **Security checklist for login cookies**

* Walk through a **real attack flow step-by-step**
* Explain **session fixation vs cookie injection**
* Show **how browsers choose cookies when duplicates exist**
* Map this to **Kubernetes / Ingress / ALB misconfigurations**
