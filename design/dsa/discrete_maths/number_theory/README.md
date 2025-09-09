Great question 👍 Let’s break **Number Theory** down in a way that makes sense both mathematically and for **competitive programming**.

---

# 🔹 What is Number Theory?

**Number Theory** is the branch of mathematics that studies the **properties and relationships of integers** (whole numbers).

It answers questions like:

* Is a number prime?
* What are its divisors?
* How do numbers behave under gcd, lcm, modular arithmetic?
* How many ways can a number be expressed as a sum or product?

Formally:

> **Number theory = the mathematics of integers and their properties (divisibility, primes, modular arithmetic, gcd/lcm, Diophantine equations, etc.)**

---

# 🔹 Key Topics in Number Theory (CP View)

| Topic                               | Description                                    | Example Problem                 |
| ----------------------------------- | ---------------------------------------------- | ------------------------------- |
| **Divisibility**                    | When does `a` divide `b`?                      | Does 12 divide 36? (Yes)        |
| **GCD & LCM**                       | Greatest common divisor, least common multiple | GCD(12,18) = 6, LCM(12,18) = 36 |
| **Prime Numbers**                   | Numbers divisible only by 1 and themselves     | Is 97 prime?                    |
| **Factorization**                   | Express number as product of primes            | 84 = 2² × 3 × 7                 |
| **Modular Arithmetic**              | Work with remainders                           | (7+5) mod 3 = 12 mod 3 = 0      |
| **Euler’s Totient φ(n)**            | Count integers ≤n coprime to n                 | φ(10) = 4                       |
| **Congruences**                     | Equations with mod                             | x ≡ 3 (mod 5)                   |
| **Chinese Remainder Theorem (CRT)** | Solve simultaneous modular equations           | x ≡ 2 mod 3, x ≡ 3 mod 5        |
| **Diophantine Equations**           | Integer solutions of equations                 | Solve 3x + 5y = 11              |
| **Modular Inverse**                 | Solve ax ≡ 1 mod m                             | Inverse of 3 mod 7 is 5         |
| **Fast Exponentiation**             | Compute a^b mod m efficiently                  | pow(a,b,m)                      |

---

# 🔹 Competitive Programming Examples

1. **GCD & LCM**

```python
import math
print(math.gcd(12, 18))  # 6
print(math.lcm(12, 18))  # 36
```

2. **Modular Exponentiation**
   (Fast way to compute a^b mod m)

```python
def mod_exp(a, b, m):
    res = 1
    a %= m
    while b > 0:
        if b % 2:
            res = (res * a) % m
        a = (a * a) % m
        b //= 2
    return res

print(mod_exp(2, 10, 1000))  # 1024 mod 1000 = 24
```

3. **Sieve of Eratosthenes** (find all primes ≤n)

```python
def sieve(n):
    prime = [True]*(n+1)
    prime[0]=prime[1]=False
    for p in range(2, int(n**0.5)+1):
        if prime[p]:
            for i in range(p*p, n+1, p):
                prime[i]=False
    return [i for i,v in enumerate(prime) if v]

print(sieve(30))  # primes up to 30
```

4. **Chinese Remainder Theorem (CRT)**
   Solve:
   x ≡ 2 mod 3
   x ≡ 3 mod 5

Answer: x = 8 (smallest solution)

---

# 🔹 Why Number Theory in CP?

* Many problems reduce to **gcd/lcm, modular arithmetic, or prime factorization**.
* Used in cryptography (RSA, modular inverses).
* Classic contest topics:

  * Modular exponentiation (fast pow)
  * Large prime checks (Miller–Rabin)
  * Counting divisors, sum of divisors
  * Modular inverses for combinatorics (nCr mod p)

---

# ✅ Summary

* **Number theory = study of integers and their properties.**
* Core ideas: **divisibility, primes, gcd/lcm, modular arithmetic, CRT, totient, Diophantine equations**.
* In competitive programming → used in **gcd/lcm problems, modular arithmetic, combinatorics mod p, prime factorization, and fast exponentiation.**

---

👉 Do you want me to build a **roadmap of number theory for CP** (basic → advanced, with Python code templates for each)?
