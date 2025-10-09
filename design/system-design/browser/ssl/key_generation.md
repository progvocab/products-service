Perfect — let’s visualize the **key generation process in TLS 1.3** at a **low-level**, focusing specifically on the **ECDHE key exchange** (using an elliptic curve like `x25519` or `secp256r1`).

Below is a **Mermaid sequence diagram** that shows step-by-step how the client and server generate and derive the **shared session key** from ephemeral keys.

---

```mermaid
sequenceDiagram
    participant Client
    participant Server

    Note over Client,Server: 🔐 TLS 1.3 - Ephemeral ECDHE Key Generation Process

    %% === STEP 1: CLIENT KEY GENERATION ===
    Client->>Client: Generate random private key (a)
    Client->>Client: Compute public key A = a * G (G = generator point)
    Client->>Server: Send "ClientHello" + KeyShare(A, curve = x25519)

    %% === STEP 2: SERVER KEY GENERATION ===
    Server->>Server: Generate random private key (b)
    Server->>Server: Compute public key B = b * G
    Server->>Client: Send "ServerHello" + KeyShare(B, curve = x25519)

    %% === STEP 3: SHARED SECRET DERIVATION ===
    Note over Client,Server: Both sides now compute shared secret using ECDHE

    Client->>Client: Compute shared_secret = a * B
    Server->>Server: Compute shared_secret = b * A
    Note over Client,Server: shared_secret = a * b * G  (identical on both sides)

    %% === STEP 4: KEY SCHEDULE USING HKDF ===
    Note over Client,Server: Apply HKDF (HMAC-based Key Derivation Function)

    Client->>Client: Derive early_secret = HKDF-Extract(0, PSK or 0)
    Client->>Client: Derive handshake_secret = HKDF-Extract(early_secret, shared_secret)
    Client->>Client: Derive master_secret = HKDF-Extract(handshake_secret, 0)

    Server->>Server: Same derivation using shared_secret

    %% === STEP 5: FINAL KEY MATERIAL ===
    Note over Client,Server: Both now have identical keys:
    Note over Client,Server:
      - handshake_traffic_keys (for handshake encryption)
      - application_traffic_keys (for data encryption)
      - finished_key (for verifying handshake integrity)

    Client->>Server: Send "Finished" (MAC over handshake)
    Server->>Client: Send "Finished" (MAC over handshake)
    Note over Client,Server: 🔑 Secure communication begins
```

---

### 🧩 Breakdown of What’s Happening

| Stage                            | Description                                                                                           | Result                                      |
| -------------------------------- | ----------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| **1. Key Generation**            | Each side generates a random private scalar and computes a public point on the chosen elliptic curve. | `(a, A)` for client and `(b, B)` for server |
| **2. Key Exchange**              | Public keys are exchanged in `ClientHello` and `ServerHello`.                                         | Both know `A` and `B`                       |
| **3. Shared Secret Computation** | Each side multiplies its private key with the other’s public key.                                     | Shared secret = `a * b * G`                 |
| **4. HKDF Key Derivation**       | The shared secret is fed into **HKDF-Extract/Expand** functions to derive handshake and session keys. | `handshake_secret`, `master_secret`         |
| **5. Traffic Keys Ready**        | Final symmetric keys used for encryption and authentication are derived.                              | Secure TLS channel established              |

---

Would you like me to extend this diagram to show **the exact HKDF-Extract and Expand stages** (e.g., how early_secret → handshake_secret → master_secret → traffic keys are chained)? That gives a truly **cryptographic-level view** of the TLS 1.3 key schedule.
