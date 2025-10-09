Let's break down both clearly — **HKDF** and **HMAC** are cryptographic primitives that appear in many modern protocols (including TLS 1.3).

---

## 🔒 **HMAC (Hash-based Message Authentication Code)**

### 🔹 Purpose

HMAC provides **message integrity and authentication** — it ensures that data hasn’t been modified and verifies who sent it.

### 🔹 How it Works

It combines:

* a **cryptographic hash function** (like SHA-256), and
* a **secret key**.

**Formula (conceptually):**
[
\text{HMAC}(K, m) = H((K' \oplus opad) | H((K' \oplus ipad) | m))
]

where

* ( H ) = hash function (e.g., SHA-256),
* ( K ) = secret key,
* ( m ) = message,
* ( opad ), ( ipad ) = fixed outer/inner padding constants.

The result is a **fixed-length MAC** that only someone with the secret key can generate or verify.

**Example use:**
TLS uses HMAC to create *Finished* messages that verify the integrity of the handshake.

---

## 🧬 **HKDF (HMAC-based Key Derivation Function)**

### 🔹 Purpose

HKDF is used to **derive strong cryptographic keys** from an initial piece of keying material (like a shared secret from Diffie-Hellman).

It doesn’t just “stretch” bits — it **extracts and expands** entropy securely using HMAC.

---

### 🔹 Two Main Steps

1. **Extract phase:**
   [
   \text{PRK} = \text{HMAC}(\text{salt}, \text{IKM})
   ]

   * `IKM` = input keying material (e.g., ECDHE shared secret)
   * `salt` = optional random value
   * `PRK` = pseudorandom key

2. **Expand phase:**
   [
   \text{OKM} = \text{HMAC}(\text{PRK}, \text{info} | 0x01)
   ]

   * `info` = context info (labels like “handshake key”)
   * `OKM` = output keying material (actual encryption keys)

---

### 🔹 Why TLS 1.3 Uses It

TLS 1.3’s entire key schedule (early secret → handshake secret → master secret → traffic keys) is built on **HKDF**, ensuring:

* Strong key separation
* Resistance to key compromise
* Simplicity: relies only on the proven security of HMAC

---

### ✅ **Summary Table**

| Function | Full Form                              | Purpose                                     | Used In                        |
| -------- | -------------------------------------- | ------------------------------------------- | ------------------------------ |
| **HMAC** | Hash-based Message Authentication Code | Authenticate data & ensure integrity        | Finished messages, record MACs |
| **HKDF** | HMAC-based Key Derivation Function     | Derive multiple keys from one shared secret | TLS 1.3 key schedule           |

---

Would you like a **Mermaid diagram** showing visually how HKDF’s *Extract → Expand* process works inside the TLS 1.3 key schedule?
