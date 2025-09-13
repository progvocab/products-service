Nice one 👌 — let’s dive into **Euclid’s law of divisors**, which is part of the foundation of number theory.

---

## 📜 Euclid’s Law of Divisors (Common Divisors Law)

It states:

> **If a number divides two numbers, it also divides their difference.**

---

### 🔹 Formal Statement

If $d \mid a$ and $d \mid b$, then:

$$
d \mid (a - b)
$$

and similarly:

$$
d \mid (a + b)
$$

---

### 🔹 Why is this important?

This property is the **basis of the Euclidean Algorithm** for computing the **GCD**.

* If you want to compute $\gcd(a, b)$, you can replace $a$ with $a-b$ (or $a \bmod b$) without changing the GCD.
* That’s because common divisors of $a$ and $b$ are the same as common divisors of $b$ and $(a-b)$.

---

### 🔹 Example

Take $a = 28$, $b = 12$.

* Common divisors of 28 and 12: $1, 2, 4$.
* $28 - 12 = 16$.
* Common divisors of 12 and 16: $1, 2, 4$.
* ✅ Same set of common divisors → law holds.

---

### 🔹 Connection to Euclidean Algorithm

The algorithm for $\gcd(a, b)$ repeatedly applies:

$$
\gcd(a, b) = \gcd(b, a \bmod b)
$$

This is **directly derived** from Euclid’s law of divisors.

---

✅ **Summary:**

* Euclid’s law of divisors: if $d$ divides both $a$ and $b$, then $d$ divides their difference (and sum).
* This principle is the **foundation of GCD computation**.

---

👉 Do you want me to also show how Euclid originally **proved** this law (from *Elements*, Book VII)?
