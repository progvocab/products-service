Perfect 👍 Modular exponentiation is **one of the most important number theory tools in competitive programming and cryptography**. Let’s go step by step.

---

# 🔹 Problem Statement

We want to compute:

$$
a^b \pmod{m}
$$

* `a` = base
* `b` = exponent (can be very large)
* `m` = modulus

---

# 🔹 Why Not Direct Computation?

Example: Compute $2^{1000} \mod 1000$

* $2^{1000}$ is **huge** (\~300 digits).
* Computing directly and then taking `% 1000` is **inefficient** and can cause overflow.

We need a **fast and memory-efficient** way.

---

# 🔹 Key Observations

1. **Mod property**

   $$
   (x \cdot y) \bmod m = ((x \bmod m) \cdot (y \bmod m)) \bmod m
   $$

   This allows us to reduce numbers step by step.

2. **Exponent Decomposition (Divide & Conquer)**

   * If $b$ is **even**:

     $$
     a^b = (a^{b/2})^2
     $$
   * If $b$ is **odd**:

     $$
     a^b = a \cdot a^{b-1}
     $$

This leads to the **Binary Exponentiation (Exponentiation by Squaring)** algorithm.

---

# 🔹 Algorithm (Binary Exponentiation)

We repeatedly square the base and halve the exponent:

* Initialize `res = 1`
* While `b > 0`:

  * If `b` is odd → multiply result by base
  * Square the base
  * Halve the exponent

---

# 🔹 Python Code

```python
def mod_exp(a, b, m):
    res = 1
    a %= m  # reduce base
    while b > 0:
        if b % 2:  # if b is odd
            res = (res * a) % m
        a = (a * a) % m  # square the base
        b //= 2          # divide exponent by 2
    return res

# Example
print(mod_exp(2, 10, 1000))  # 1024 mod 1000 = 24
```

---

# 🔹 Step-by-Step Example

Compute $3^{13} \mod 7$

1. `res = 1, a = 3, b = 13`
2. `b=13 (odd)` → res = 1×3 %7 = 3, a=3²=9%7=2, b=6
3. `b=6 (even)` → res=3, a=2²=4%7=4, b=3
4. `b=3 (odd)` → res=3×4%7=5, a=4²=16%7=2, b=1
5. `b=1 (odd)` → res=5×2%7=3, a=2²=4, b=0

✅ Final Answer = **3**

---

# 🔹 Time Complexity

* Naive: $O(b)$ multiplications
* Binary Exponentiation: $O(\log b)$

For very large `b` (like $10^{18}$), this makes problems solvable.

---

# 🔹 Applications in CP & Cryptography

1. **Fast power computations**

   * Computing powers mod p
   * Large factorial computations mod p

2. **Modular inverse (when modulus is prime)**

   * Fermat’s little theorem:

     $$
     a^{p-2} \pmod{p} = a^{-1} \pmod{p}
     $$

3. **RSA encryption/decryption**

   * Encryption: $c = m^e \mod n$
   * Decryption: $m = c^d \mod n$

4. **Discrete logarithm problems**

   * Used in Diffie–Hellman key exchange

---

✅ So, **modular exponentiation is the go-to method for handling huge exponents under a modulus efficiently.**

---

Would you like me to also explain **how Python’s built-in `pow(a, b, m)` works internally** (it actually implements this algorithm)?

