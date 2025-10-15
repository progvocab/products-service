# **TLS 1.2 (RSA key exchange) handshake** flow 

### Prerequisite
Server to generate Key Pairs and Certificate

```mermaid
sequenceDiagram
    participant Server
    participant CA as Certificate Authority

    Note over Server,CA: **Prerequisites**
    Server->>Server: Generate long-term RSA key pair (Private, Public)
    Server->>CA: Submit CSR (Certificate Signing Request)
    CA-->>Server: Issue signed certificate (includes Server's Public Key)
    Server->>CA: Upload certificate to CA store (for distribution)
```

### DNS Lookup
Recursive DNS Lookup

### TCP Handshake
TCP Handshake SYN , SYN - ACK , ACK

### TLS Handshake

```mermaid
sequenceDiagram
    %% =============================
    %% TLS 1.2 HANDSHAKE (RSA MODE)
    %% =============================

    participant Client
    participant Server
    
    

    Note over Client,Server: **TLS Handshake Begins**

    Client->>Server: (1) ClientHello<br/>TLS Versions: 1.1, 1.2, 1.3<br/>Cipher Suites:<br/>TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256<br/>TLS_DHE_RSA_WITH_AES_128_GCM_SHA256<br/>TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256<br/>TLS_RSA_WITH_AES_128_GCM_SHA256<br/>TLS_RSA_WITH_AES_256_CBC_SHA<br/>TLS_RSA_WITH_3DES_EDE_CBC_SHA<br/>client_random
    Server-->>Client: (2) ServerHello<br/>Chosen: TLS 1.2<br/>Cipher Suite: TLS_RSA_WITH_AES_128_GCM_SHA256<br/>server_random
    Server-->>Client: (3) Certificate Chain<br/>(Leaf, Intermediate, Root)

    Note over Client: **Certificate Validation**
    Client->>Client: (4) Validate leaf certificate (extract Server’s Public Key)
    Client->>Client: (5) Validate intermediate certificate
    Client->>Client: (6) Validate root certificate against local trust store<br/>Verify hostname/IP

    Note over Client,Server: **Key Exchange Phase**

    Client->>Client: (7) Generate 48-byte Pre-Master Secret using CSPRNG (os.urandom)
    Client->>Server: Encrypt Pre-Master Secret with Server’s RSA Public Key
    Server->>Server: (8) Decrypt Pre-Master Secret using RSA Private Key

    Note over Client,Server: **Derive Keys**
    Client->>Client: (9–10) Compute Master Secret:<br/>PRF(pre_master_secret, "master secret", client_random + server_random)
    Client->>Client: Derive Session Keys:<br/>PRF(master_secret, "key expansion", server_random + client_random)
    Server->>Server: (9–10) Same derivation (identical results)

    Note over Client,Server: **Session Keys Created**
    Client->>Client: client_write_key, client_MAC_key, IV
    Server->>Server: server_write_key, server_MAC_key, IV

    Note over Client,Server: **Finalize Handshake**
    Client->>Server: (12) ChangeCipherSpec → Switch to encrypted mode
    Client->>Server: (13) Finished (encrypted)
    Server->>Client: (14) ChangeCipherSpec → Switch to encrypted mode
    Server->>Client: (15) Finished (encrypted)

    Note over Client,Server: **Secure Communication Begins**

    Client->>Server: (16) HTTPS Request<br/>(Encrypted with session key)
    Server->>Server: Decrypt using session key
    Server-->>Client: (17) HTTPS Response<br/>(Encrypted with session key)
    Client->>Client: Decrypt response using session key

    Note over Client,Server: (18) HTTPS stateless operation completed
```


### Notes

* **Pre-master secret** is encrypted using the server’s **long-term RSA public key**.
* **Master secret** is derived using PRF (HMAC-based function).
* **Session keys** (client/server write keys, MAC keys, IVs) are derived separately.
* **ChangeCipherSpec** transitions both ends to encrypted mode.
* **Finished** messages confirm both sides derived identical keys.



