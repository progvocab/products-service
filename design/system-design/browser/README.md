# **Modern Browser** 

### High Level Components

* **User Interface Layer** (address bar, tabs, buttons, bookmarks)
* **Browser Engine** (manages communication between UI and rendering engine)
* **Rendering Engine** (parses HTML/CSS, builds DOM + CSSOM, renders layout + painting)
* **JavaScript Engine** (parses and executes JS, e.g. V8 in Chrome, JavaScriptCore in Safari)
* **Networking** (handles HTTP/HTTPS, caching, cookies, proxy, TLS/SSL)
* **Data Storage** (cookies, localStorage, IndexedDB, cache, service workers)
* **UI Backend** (compositing, rendering graphics, GPU acceleration)



```mermaid
flowchart TD
    subgraph UI["User Interface"]
        A[Address Bar]
        B[Back/Forward, Refresh]
        C[Tabs & Window Manager]
    end

    subgraph BrowserEngine["Browser Engine"]
        D[Controller] --> E[Communicator]
    end

    subgraph RenderingEngine["Rendering Engine"]
        F[HTML Parser] --> G[DOM Tree]
        H[CSS Parser] --> I[CSSOM Tree]
        G --> J[Render Tree]
        I --> J
        J --> K[Layout Engine]
        K --> L[Painting & Compositing]
    end

    subgraph JavaScriptEngine["JavaScript Engine"]
        M[Parser]
        N[Interpreter / JIT Compiler]
        O[Execution Contexts]
    end

    subgraph Networking["Networking Layer"]
        P[HTTP/HTTPS Requests]
        Q[Cache, Cookies, Proxy]
        R[TLS/SSL Handling]
    end

    subgraph Storage["Persistent Storage"]
        S[Cookies]
        T[LocalStorage / SessionStorage]
        U[IndexedDB]
        V[Cache Storage]
    end

    subgraph Backend["UI Backend & GPU"]
        W[Graphics Rendering]
        X[GPU Acceleration]
    end

    %% Connections
    A --> D
    B --> D
    C --> D
    D --> F
    D --> H
    D --> M
    D --> P
    F --> G
    H --> I
    M --> N --> O --> J
    P --> Q --> R --> F
    P --> Q --> H
    J --> K --> L --> W --> X
    L --> UI
    Q --> S
    Q --> T
    Q --> U
    Q --> V
```

---

###  Flow :

1. **UI Layer** → User interacts (type URL, click buttons).
2. **Browser Engine** → Acts as the coordinator, passing tasks to the rendering engine or JS engine.
3. **Rendering Engine**

   * Parses HTML into **DOM**.
   * Parses CSS into **CSSOM**.
   * Combines them into a **Render Tree**, then calculates **layout** and **paints** pixels.
4. **JavaScript Engine**

   * Parses and executes JS code.
   * Updates the DOM/CSSOM dynamically (causing re-render).
5. **Networking**

   * Manages requests (HTTP/HTTPS), caching, cookies, and SSL.
6. **Storage**

   * Manages local data like cookies, localStorage, IndexedDB.
7. **UI Backend + GPU**

   * Handles actual rendering of graphics, fonts, images, accelerated by GPU.

---

** flow when a user types a URL like `google.com`** into a browser
- DNS resolution
- TCP/TLS handshake
- HTTP request/response
- Rendering



```mermaid
flowchart TD
    A[User types URL: google.com] --> B[Browser checks cache for DNS entry]
    
    B -->|Cache miss| C[OS checks local DNS cache]
    C -->|Cache miss| D[Recursive DNS Resolver]
    D --> E[Root DNS Server]
    E --> F[TLD DNS Server  .com ]
    F --> G[Authoritative DNS Server for google.com]
    G --> H[Return IP to browser via resolver]
    
    H --> I[Browser initiates TCP connection to server IP]
    I --> J[TCP 3-way handshake completes]
    
    J --> K[Browser initiates TLS handshake for HTTPS ]
    K --> L[TLS session established]
    
    L --> M[Browser sends HTTP GET request to server]
    M --> N[Server processes request]
    N --> O[Server responds with HTML, CSS, JS, Images]
    
    O --> P[Browser parses HTML]
    P --> Q[Builds DOM tree]
    O --> R[Parses CSS → CSSOM tree]
    Q & R --> S[Construct Render Tree]
    S --> T[Layout calculation]
    T --> U[Painting & Compositing]
    U --> V[Home page displayed to user]
    
    %% Optional: Browser cache for resources
    O --> W[Cache static resources for future requests]
```

---

### 🔹 Flow Explanation:

1. **DNS Resolution**

   * Browser → OS → Recursive DNS → Root → TLD → Authoritative DNS → IP address.

2. **TCP Connection**

   * Browser establishes a TCP connection (3-way handshake) to the server.
- Client SYN 
- Server SYN + ACK
- Client ACK

3. **TLS/SSL Handshake**

   * Browser and server negotiate encryption (HTTPS).
- Client Hello : sharing cipher suites , protocol 
- generate random private key , share it's public key
- Server Hello : selecting cipher suite , protocol,  sharing certificate , confirming the elliptic curve to use ( TLS1.3)
- generate random private key , share it's public key
- Client: Validating Left Certificate-> Intermediatory Certificate -> Root Certificate using client's CA file 
- validating the domain name is owned by server host
- Generate the common points on the elliptic curve and creating a shared secret by both server and client.
- Server : creating ephemeral key pair random , signing the ephemeral public key using long term private key

4. **HTTP Request**

   * Browser sends GET request to fetch homepage.

5. **Server Response**

   * Server sends HTML, CSS, JS, images, etc.

6. **Rendering**

   * Browser parses HTML → DOM tree.
   * Parses CSS → CSSOM tree.
   * Combines to Render Tree → layout → painting → display.

7. **Caching**

   * Browser caches static resources for faster subsequent loads.

---

If you want, I can also **draw a more detailed diagram showing parallel requests for images, JS, CSS, and asynchronous loading of resources**, which happens in modern browsers for sites like Google.



