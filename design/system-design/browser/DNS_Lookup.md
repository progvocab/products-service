

When you type a URL (say `www.google.com`) in the browser, and the **OS cache doesn’t have the DNS → IP mapping**, the system goes through the full DNS resolution flow.

---

### 🔹 Steps Involved (DNS Lookup without OS Cache)

1. **Browser Check** – Browser checks its own DNS cache.
2. **OS Cache Check** – OS checks local DNS resolver cache (empty in this case).
3. **DNS Resolver (usually ISP/Configured DNS)** – Request is forwarded to the recursive DNS resolver.
4. **Root DNS Server** – Resolver queries root server to find the responsible TLD server (e.g., `.com`).
5. **TLD DNS Server** – TLD server responds with the authoritative DNS server for the domain (`google.com`).
6. **Authoritative DNS Server** – Resolver queries it to get the actual IP address of `www.google.com`.
7. **Resolver Caching** – Resolver caches the response for future use.
8. **OS Caching** – OS caches the IP for TTL (time-to-live).
9. **Browser Proceeds** – Browser uses the IP to initiate TCP/TLS connection with the server.

---

### 🔹 Mermaid Diagram

```mermaid
sequenceDiagram
    participant User as User (Browser)
    participant Browser as Browser Cache
    participant OS as OS DNS Cache
    participant Resolver as Recursive DNS Resolver (ISP/Configured)
    participant Root as Root DNS Server
    participant TLD as TLD DNS Server (.com)
    participant Auth as Authoritative DNS Server
    participant Server as Target Web Server

    User->>Browser: Request www.google.com
    Browser->>OS: Check DNS Cache
    OS-->>Browser: Cache Miss (No IP)
    Browser->>Resolver: DNS Query www.google.com
    Resolver->>Root: Query Root Server
    Root-->>Resolver: Return TLD Server (.com)
    Resolver->>TLD: Query TLD Server
    TLD-->>Resolver: Return Authoritative DNS Server
    Resolver->>Auth: Query Authoritative DNS Server
    Auth-->>Resolver: Return IP Address
    Resolver-->>OS: Cache + Return IP Address
    OS-->>Browser: Pass IP Address
    Browser->>Server: Initiate TCP/TLS Connection with IP
```

---

Would you like me to also extend this diagram to **show caching hierarchy visually** (Browser → OS → Resolver → Root → TLD → Authoritative) as a **flowchart-style diagram**, or keep it strictly sequence-based?
