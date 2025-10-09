Let’s walk through **how the client encrypts data using the derived key material** in TLS 1.3 — this is where the handshake’s math turns into actual message protection 🔐

---

## 🧩 1. Key Material Recap

After the TLS 1.3 handshake, both **client** and **server** derive the same symmetric keys using **HKDF**:

| Key Type                            | Direction       | Purpose                         |
| ----------------------------------- | --------------- | ------------------------------- |
| `client_application_traffic_secret` | Client → Server | Encrypt app data sent by client |
| `server_application_traffic_secret` | Server → Client | Encrypt app data sent by server |

Each secret is expanded into:

* **AES or ChaCha20 encryption key**
* **Initialization Vector (IV)** for the cipher

---

## ⚙️ 2. Encryption Algorithm

TLS 1.3 uses **Authenticated Encryption with Associated Data (AEAD)** ciphers — like:

* `AES_128_GCM`
* `AES_256_GCM`
* `ChaCha20_Poly1305`

These encrypt **and authenticate** data in one operation.

---

## 🔐 3. Encryption Process (Client → Server)

Let’s break down what happens when the client sends its first encrypted record:

### Step 1: Prepare the plaintext

For example:

```
HTTP Request: "GET /index.html HTTP/1.1\r\nHost: example.com\r\n\r\n"
```

### Step 2: Compute the per-record nonce

Each record uses a unique nonce derived from:

```
nonce = IV XOR record_sequence_number
```

This ensures every TLS record gets a unique initialization vector — crucial for GCM security.

### Step 3: Encrypt with AEAD

```
ciphertext, auth_tag = AEAD_Encrypt(
    key = client_write_key,
    nonce = computed_nonce,
    plaintext = application_data,
    AAD = record_header
)
```

* `AAD` (Associated Authenticated Data): the record header, which is not encrypted but authenticated.
* `auth_tag`: authentication tag ensuring data integrity.

### Step 4: Send the encrypted record

The client sends:

```
TLSCiphertext = [Record Header] + [Ciphertext] + [Auth Tag]
```

---

## 🔁 4. Decryption on Server Side

When the server receives the record:

1. It computes the same nonce.
2. Uses the same AEAD cipher and key:

   ```
   plaintext = AEAD_Decrypt(key, nonce, ciphertext, AAD)
   ```
3. Verifies the `auth_tag`.
   If verification fails → **connection is terminated**.

---

## ✅ 5. Summary

| Step | Operation                 | Description                        |
| ---- | ------------------------- | ---------------------------------- |
| 1    | Derive symmetric key & IV | From traffic secret using HKDF     |
| 2    | Compute nonce             | IV XOR sequence number             |
| 3    | Encrypt data              | AEAD (AES-GCM / ChaCha20-Poly1305) |
| 4    | Add auth tag              | Ensures integrity                  |
| 5    | Send ciphertext           | Server decrypts symmetrically      |

---

Would you like me to draw a **Mermaid diagram** of this encryption–decryption flow (showing the nonce derivation, AEAD encryption, and transmission steps)? It visually ties the key material to actual TLS record encryption.
