Excellent — this question goes straight to the **core of how TLS trust works.**

Let’s unpack it carefully — starting with the intuition, then the technical purpose, and finally how it’s used in real TLS handshakes.

---

## 🔍 The Core Idea

When a client (like a browser or CLI HTTP client) connects to a server over **HTTPS (TLS)**,
it needs to **verify that the server is who it claims to be.**

But the client doesn’t automatically “trust” every server certificate it sees — it only trusts certificates **signed by a trusted Certificate Authority (CA)**.

That’s where the **certificate chain** comes in.

---

## 🧩 What Is the Certificate Chain

The **certificate chain** (or “chain of trust”) is the sequence of certificates that connect the **server’s certificate** to a **trusted root certificate authority**.

Example chain:

```text
[1] Server Certificate: example.com
        ⬇ signed by
[2] Intermediate CA: Let's Encrypt Authority X3
        ⬇ signed by
[3] Root CA: ISRG Root X1 (trusted by client)
```

When the server sends this **chain** to the client during the TLS handshake,
the client can verify signatures step by step until it reaches a **root CA it already trusts** (in `/etc/ssl/cert.pem`).

---

## 🎯 Purpose of Sending the Certificate Chain

| #     | **Purpose**                   | **Explanation**                                                                                                                          |
| ----- | ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| **1** | 🔐 **Authentication**         | Proves that the server really owns the domain name (via the Common Name or SAN field).                                                   |
| **2** | 🧱 **Build the trust path**   | The client may not have the intermediate CA certificates locally — sending the chain helps it verify the full path up to a trusted root. |
| **3** | 🧾 **Signature verification** | Each certificate is digitally signed by its issuer; the client validates these signatures recursively.                                   |
| **4** | 🧩 **Enable compatibility**   | Not all clients have the same set of intermediates — the server provides them to ensure successful verification on older systems.        |
| **5** | ⚡ **Optimize handshake**      | Sending the full chain avoids extra network requests to fetch intermediate CAs (which would slow down the TLS setup).                    |

---

## 🧠 How the Client Uses the Chain

Here’s what happens step-by-step:

1. **Server sends chain** (leaf + intermediates).
2. **Client checks:**

   * Is the **leaf certificate** valid for the domain (SAN / CN)?
   * Was it **signed** by the next certificate in the chain?
   * Is each certificate in the chain **valid and not expired**?
   * Does the **root CA** appear in the client’s local trust store (e.g., `/etc/ssl/cert.pem`)?
3. If all checks pass, trust is established —
   and encryption keys are exchanged securely.

If **any certificate in the chain fails verification**, the TLS connection is aborted with an error like:

```
SSL_ERROR_BAD_CERT_DOMAIN
ERR_CERT_AUTHORITY_INVALID
ERR_CERT_DATE_INVALID
```

---

## 🧩 Example Chain in Action

When connecting to `https://example.com`, the server might send:

```
-----BEGIN CERTIFICATE-----
(Server Certificate)
CN = example.com
Issuer = Let's Encrypt R3
-----END CERTIFICATE-----

-----BEGIN CERTIFICATE-----
(Intermediate)
CN = Let's Encrypt R3
Issuer = ISRG Root X1
-----END CERTIFICATE-----
```

Your client already has **ISRG Root X1** in its trust store —
so it verifies the chain like:

```
example.com (leaf)
  ⬇ signed by
Let's Encrypt R3 (intermediate)
  ⬇ signed by
ISRG Root X1 (root, trusted)
```

✅ Connection is trusted.

---

## 🧰 Without the Chain (What Would Go Wrong)

If the server only sent its **own** certificate:

* The client might not know which CA signed it.
* It may not have the intermediate CA locally.
* Verification would fail →
  `"certificate signed by unknown authority"`

That’s why the **chain** must be sent — it completes the trust path.

---

## 🔐 TLS 1.3 Note

In **TLS 1.3**, the “Certificate” message still contains the full chain (except the root).
It’s typically compressed or optimized but semantically the same:
the client still validates from **leaf → intermediate → root**.

---

## 📘 Summary

| Concept                | Explanation                                                                     |
| ---------------------- | ------------------------------------------------------------------------------- |
| **Certificate Chain**  | Sequence of certificates connecting the server’s cert to a trusted root         |
| **Purpose**            | Prove server identity and establish trust                                       |
| **Why Sent by Server** | Client may not have intermediates; sending them ensures full trust verification |
| **Verification Path**  | Leaf → Intermediate(s) → Root (trusted locally)                                 |
| **Failure Case**       | Missing or invalid intermediate → SSL verification error                        |

---

Would you like me to show this with a **Mermaid chain-of-trust diagram** showing how the client validates each certificate using public keys and signatures?
