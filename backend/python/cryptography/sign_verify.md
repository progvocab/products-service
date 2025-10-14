Perfect — let’s do both parts:

✅ **Sign & verify** (works with both Ed25519 and RSA)
✅ **Encrypt & decrypt** (only for RSA — Ed25519 doesn’t support encryption, it’s purely for signatures)

Below is a complete, clear Python example using the `cryptography` library 👇

---

## 🧩 **Full Example: Sign / Verify / Encrypt / Decrypt**

```python
from cryptography.hazmat.primitives.asymmetric import ed25519, rsa, padding
from cryptography.hazmat.primitives import hashes, serialization

# ========== 1️⃣ Generate keys ==========
# --- Ed25519 key pair ---
ed_private_key = ed25519.Ed25519PrivateKey.generate()
ed_public_key = ed_private_key.public_key()

# --- RSA key pair (2048-bit) ---
rsa_private_key = rsa.generate_private_key(
    public_exponent=65537,
    key_size=2048
)
rsa_public_key = rsa_private_key.public_key()

message = b"Confidential message for signature and encryption."

# ========== 2️⃣ Ed25519 Sign & Verify ==========
signature = ed_private_key.sign(message)
print("✅ Ed25519 signature created.")

try:
    ed_public_key.verify(signature, message)
    print("✅ Ed25519 signature verified successfully.")
except Exception as e:
    print("❌ Verification failed:", e)


# ========== 3️⃣ RSA Sign & Verify ==========
rsa_signature = rsa_private_key.sign(
    message,
    padding.PSS(
        mgf=padding.MGF1(hashes.SHA256()),
        salt_length=padding.PSS.MAX_LENGTH
    ),
    hashes.SHA256()
)
print("✅ RSA signature created.")

try:
    rsa_public_key.verify(
        rsa_signature,
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256()
    )
    print("✅ RSA signature verified successfully.")
except Exception as e:
    print("❌ RSA verification failed:", e)


# ========== 4️⃣ RSA Encrypt & Decrypt ==========
ciphertext = rsa_public_key.encrypt(
    message,
    padding.OAEP(
        mgf=padding.MGF1(algorithm=hashes.SHA256()),
        algorithm=hashes.SHA256(),
        label=None
    )
)
print("✅ RSA encryption complete.")

plaintext = rsa_private_key.decrypt(
    ciphertext,
    padding.OAEP(
        mgf=padding.MGF1(algorithm=hashes.SHA256()),
        algorithm=hashes.SHA256(),
        label=None
    )
)
print("✅ RSA decryption complete.")
print("🔓 Decrypted message:", plaintext.decode())
```

---

## ⚙️ **Install Dependencies**

```bash
pip install cryptography
```

---

## 📘 **Output Example**

```
✅ Ed25519 signature created.
✅ Ed25519 signature verified successfully.
✅ RSA signature created.
✅ RSA signature verified successfully.
✅ RSA encryption complete.
✅ RSA decryption complete.
🔓 Decrypted message: Confidential message for signature and encryption.
```

---

## 🧠 Notes

| Algorithm   | Purpose                                       | Supports Encryption? | Supports Signature? |
| ----------- | --------------------------------------------- | -------------------- | ------------------- |
| **Ed25519** | Modern elliptic-curve signing (fast, compact) | ❌                    | ✅                   |
| **RSA**     | General-purpose asymmetric crypto             | ✅                    | ✅                   |

* **Ed25519** → used in SSH, Git, and modern authentication.
* **RSA** → used in TLS certificates, JWT signing, hybrid encryption.

---

Would you like me to extend this to show **hybrid encryption** (using RSA to encrypt a symmetric AES key, which encrypts large files or data)?
