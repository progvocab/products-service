Perfect — you’re diving deep into the TLS handshake 👏

Let’s break down **what exactly happens when the server sends its certificate** during the **Server Hello** phase — and what each component inside that certificate means.

---

## 🧭 Where It Fits in the TLS Handshake

Let’s first put it in context.

### Simplified TLS 1.2 / TLS 1.3 Handshake (overview):

```text
Client                         Server
  |  ---- ClientHello -------->  |
  |  <--- ServerHello ---------- |
  |  <--- Certificate -----------|
  |  <--- ServerKeyExchange ---- |
  |  <--- ServerHelloDone -------|
  |  ---- ClientKeyExchange ---> |
  |  ---- ChangeCipherSpec ----> |
  |  ---- Finished -------------> |
  |  <--- ChangeCipherSpec ------|
  |  <--- Finished --------------|
```

👉 The **Certificate** message comes **right after ServerHello**, and it contains **the server’s identity** and **public key information**.

---

## 🧩 Components of the Certificate (sent by the Server)

The server sends **a chain of certificates**, usually:

1. **Leaf Certificate** – the website’s own certificate (`example.com`)
2. **Intermediate CA Certificate(s)** – issued by a trusted CA
3. *(Optional)* Root CA – often omitted because the client already has it in `/etc/ssl/cert.pem`

Each certificate in this chain follows **X.509** standard format.

---

### 🧱 Structure of an X.509 Certificate

Below are the **main fields** you’ll find in the server certificate:

| **Component**               | **Description**                                     | **Example / Value**                                |
| --------------------------- | --------------------------------------------------- | -------------------------------------------------- |
| **Version**                 | Version of X.509 standard (usually v3)              | `Version 3 (0x2)`                                  |
| **Serial Number**           | Unique ID assigned by issuing CA                    | `0x03b3f2c9c9d9e2a8`                               |
| **Signature Algorithm**     | Algorithm used by CA to sign this certificate       | `sha256WithRSAEncryption`                          |
| **Issuer**                  | The CA that issued this certificate                 | `CN=Let's Encrypt Authority X3, O=Let's Encrypt`   |
| **Validity Period**         | The start and end date of certificate validity      | `Not Before: 2025-10-01` → `Not After: 2026-01-01` |
| **Subject**                 | The entity the certificate belongs to (the website) | `CN=example.com, O=Example Inc, C=US`              |
| **Subject Public Key Info** | Contains the public key + algorithm used            | Public key (RSA/ECDSA)                             |
| **Extensions**              | Extra metadata and constraints                      | See below ↓                                        |

---

### 🔍 Common **Extensions** in Server Certificates

| **Extension**                          | **Purpose**                                     | **Example**                              |
| -------------------------------------- | ----------------------------------------------- | ---------------------------------------- |
| **Subject Alternative Name (SAN)**     | Lists all domains this certificate is valid for | `DNS:example.com, DNS:www.example.com`   |
| **Key Usage**                          | What operations this key can perform            | `Digital Signature, Key Encipherment`    |
| **Extended Key Usage**                 | Higher-level purpose                            | `TLS Web Server Authentication`          |
| **Basic Constraints**                  | Defines whether it’s a CA or end-entity cert    | `CA:FALSE`                               |
| **Authority Key Identifier**           | Links to the issuer’s public key                | `keyid:12:AB:34:CD...`                   |
| **Subject Key Identifier**             | Identifies the key that belongs to this cert    | `keyid:56:78:90:EF...`                   |
| **CRL Distribution Points**            | Where to find revocation lists                  | `http://crl.letsencrypt.org/...`         |
| **Authority Information Access (AIA)** | Info about the issuer CA and OCSP responder     | `OCSP - URI:http://ocsp.letsencrypt.org` |

---

## ⚙️ Example — Viewing a Real Server Certificate

You can see all this with `openssl`:

```bash
openssl s_client -connect example.com:443 -showcerts
```

or view just the structure:

```bash
openssl x509 -in server.crt -text -noout
```

Sample output snippet:

```text
Certificate:
    Data:
        Version: 3 (0x2)
        Serial Number: 0x03b3f2c9c9d9e2a8
        Signature Algorithm: sha256WithRSAEncryption
        Issuer: CN=Let's Encrypt Authority X3, O=Let's Encrypt
        Validity
            Not Before: Oct  1 00:00:00 2025 GMT
            Not After : Jan  1 00:00:00 2026 GMT
        Subject: CN=example.com
        Subject Public Key Info:
            Public Key Algorithm: rsaEncryption
                Public-Key: (2048 bit)
```

---

## 🔐 Purpose of the Server Certificate in TLS

| **Goal**                  | **Description**                                |
| ------------------------- | ---------------------------------------------- |
| **Authenticate Identity** | Prove that the server really is `example.com`  |
| **Enable Encryption**     | Provide the public key for secure key exchange |
| **Build Trust Chain**     | Allow the client to verify via CA hierarchy    |
| **Prevent MITM**          | Protects against man-in-the-middle attacks     |

---

## 🧠 TLS 1.3 Difference

In **TLS 1.3**, the server still sends a certificate, but:

* The **ServerHello** and **Certificate** messages are **encrypted earlier** in the handshake.
* Fewer round trips are required.
* The **CertificateRequest** (for mutual TLS) is optional but standardized.

---

Would you like me to draw a **Mermaid diagram** showing the exact packet flow of the **TLS handshake with certificate exchange** (TLS 1.2 and 1.3 side by side)?
