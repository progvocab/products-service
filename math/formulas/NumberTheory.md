 here’s a **complete Number Theory Formula Handbook** in Markdown, covering everything you’d ever need in **competitive programming**.

---

# 📘 Number Theory Formulas for Competitive Programming

---

## 🔹 1. Divisibility & Factors

* **Divisibility Rule**:
  $a \mid b \iff \exists k \in \mathbb{Z}, b = a \cdot k$

* **Number of divisors** (if $n = p_1^{e_1} p_2^{e_2} \dots p_k^{e_k}$):

$$
d(n) = (e_1+1)(e_2+1)\dots(e_k+1)
$$

* **Sum of divisors**:

$$
\sigma(n) = \prod_{i=1}^k \frac{p_i^{e_i+1}-1}{p_i-1}
$$

* **Perfect numbers**: $n$ is perfect if $\sigma(n) = 2n$.

---

## 🔹 2. GCD & LCM

* Euclidean Algorithm:

$$
\gcd(a,b) = \gcd(b, a \bmod b)
$$

* Relationship:

$$
\text{lcm}(a,b) \cdot \gcd(a,b) = a \cdot b
$$

---

## 🔹 3. Modular Arithmetic

* Congruence:

$$
a \equiv b \pmod{m} \iff m \mid (a-b)
$$

* Modular addition/multiplication:

$$
(a+b) \bmod m = (a \bmod m + b \bmod m) \bmod m
$$

$$
(a\cdot b) \bmod m = ((a \bmod m)\cdot(b \bmod m)) \bmod m
$$

* Modular exponentiation (fast power):

$$
a^b \bmod m
$$

---

## 🔹 4. Modular Inverses

* If $\gcd(a,m)=1$, then $a$ has a modular inverse.

* Using **Extended Euclidean Algorithm**:

$$
ax \equiv 1 \pmod{m}
$$

* Using **Fermat’s theorem** (when $m$ is prime):

$$
a^{-1} \equiv a^{m-2} \pmod{m}
$$

---

## 🔹 5. Prime Numbers

* Prime number theorem:

$$
\pi(n) \sim \frac{n}{\ln n}
$$

* Sieve of Eratosthenes complexity: $O(n \log \log n)$.

* Wilson’s Theorem:

$$
(p-1)! \equiv -1 \pmod{p}, \quad p \text{ prime}
$$

---

## 🔹 6. Euler’s Totient Function

* Definition: number of integers $\leq n$ coprime to $n$.

* Formula (if $n = p_1^{e_1}p_2^{e_2}\dots p_k^{e_k}$):

$$
\phi(n) = n \prod_{i=1}^k \left(1 - \frac{1}{p_i}\right)
$$

* Properties:

$$
\sum_{d|n} \phi(d) = n
$$

$$
a^{\phi(n)} \equiv 1 \pmod{n}, \quad \gcd(a,n)=1 \quad (\text{Euler’s theorem})
$$

---

## 🔹 7. Fermat’s Little Theorem

* If $p$ prime and $\gcd(a,p)=1$:

$$
a^{p-1} \equiv 1 \pmod{p}
$$

---

## 🔹 8. Chinese Remainder Theorem (CRT)

* System of congruences:

$$
x \equiv a_i \pmod{m_i}, \quad \gcd(m_i, m_j) = 1
$$

Unique solution modulo:

$$
M = m_1 m_2 \dots m_k
$$

---

## 🔹 9. Factorials & Binomial Coefficients

* Factorial: $n! = 1\cdot2\cdot3\cdots n$

* Binomial coefficient:

$$
\binom{n}{r} = \frac{n!}{r!(n-r)!}
$$

* Modular binomial coefficient (prime $p$):

$$
\binom{n}{r} \bmod p = \frac{n!}{r!(n-r)!} \bmod p
$$

* Lucas Theorem (binomial mod prime):

$$
\binom{n}{r} \bmod p = \prod \binom{n_i}{r_i} \bmod p
$$

(where $n_i, r_i$ are digits of $n, r$ in base $p$).

---

## 🔹 10. Quadratic Residues

* Legendre Symbol:

$$
\left(\frac{a}{p}\right) = \begin{cases}
0 & p \mid a \\
1 & a \text{ quadratic residue mod } p \\
-1 & \text{otherwise}
\end{cases}
$$

* Euler’s Criterion:

$$
\left(\frac{a}{p}\right) \equiv a^{\frac{p-1}{2}} \pmod{p}
$$

---

## 🔹 11. Diophantine Equations

* Linear Diophantine equation:

$$
ax + by = c \quad \text{has solution iff } \gcd(a,b) \mid c
$$

---

## 🔹 12. Special Sequences

* Fibonacci modulo $m$ (Pisano period).
* Catalan number (combinatorial applications):

$$
C_n = \frac{1}{n+1}\binom{2n}{n}
$$

---

## 🔹 13. Approximations

* Stirling’s Approximation:

$$
n! \approx \sqrt{2\pi n}\left(\frac{n}{e}\right)^n
$$

* Harmonic number:

$$
H_n = 1 + \frac{1}{2} + \dots + \frac{1}{n} \approx \ln n + \gamma
$$

---

✅ These are the **most important number theory formulas** used in contests — covering divisibility, modular arithmetic, primes, Euler/CRT, factorials, quadratic residues, and Diophantine equations.

---

Would you like me to now **add Python code templates** (like `gcd`, `mod_inverse`, `nCr mod p`, `sieve`, `CRT solver`) alongside each formula so you get both **math + implementation** in one cheat sheet?
