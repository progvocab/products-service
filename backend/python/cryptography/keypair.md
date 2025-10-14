Here’s a clean, working **Python example** to generate **key pairs** using both **Ed25519** and **RSA**, depending on what you choose 👇

---

### 🧩 **Option 1 — Generate Ed25519 Key Pair**

```python
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

# Generate private key
private_key = ed25519.Ed25519PrivateKey.generate()

# Get public key
public_key = private_key.public_key()

# Serialize private key (PEM format)
pem_private = private_key.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption()
)

# Serialize public key (PEM format)
pem_public = public_key.public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo
)

# Write to files
with open("ed25519_private_key.pem", "wb") as f:
    f.write(pem_private)

with open("ed25519_public_key.pem", "wb") as f:
    f.write(pem_public)

print("✅ Ed25519 key pair generated successfully!")
```

---

### 🧩 **Option 2 — Generate RSA Key Pair**

```python
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization, hashes

# Generate RSA private key (2048-bit)
private_key = rsa.generate_private_key(
    public_exponent=65537,
    key_size=2048
)

# Get public key
public_key = private_key.public_key()

# Serialize private key (PEM format)
pem_private = private_key.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption()
)

# Serialize public key (PEM format)
pem_public = public_key.public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo
)

# Write to files
with open("rsa_private_key.pem", "wb") as f:
    f.write(pem_private)

with open("rsa_public_key.pem", "wb") as f:
    f.write(pem_public)

print("✅ RSA key pair generated successfully!")
```

---

### ⚙️ **Requirements**

Install the `cryptography` library if you haven’t already:

```bash
pip install cryptography
```

---

### 📂 Output

This will generate two PEM files in your working directory:

| File                                              | Contains    | Format                   |
| ------------------------------------------------- | ----------- | ------------------------ |
| `ed25519_private_key.pem` / `rsa_private_key.pem` | Private key | PKCS#8 PEM               |
| `ed25519_public_key.pem` / `rsa_public_key.pem`   | Public key  | SubjectPublicKeyInfo PEM |

---

Would you like me to extend this example to **sign and verify a message** using the generated keys (so you can test they work)?
