 in competitive programming, besides **combinatorics & probability**, there’s a whole set of **must-know math formulas** that save you from re-deriving things under time pressure. Let’s build a **master formula sheet**.

---

# 📘 Important Formulas for Competitive Programming

---

## 🔹 1. Arithmetic & Geometric Progressions

* Sum of first $n$ natural numbers:

$$
S = \frac{n(n+1)}{2}
$$

* Sum of squares:

$$
S = \frac{n(n+1)(2n+1)}{6}
$$

* Sum of cubes:

$$
S = \left(\frac{n(n+1)}{2}\right)^2
$$

* Arithmetic progression (AP):

$$
S_n = \frac{n}{2}[2a + (n-1)d]
$$

* Geometric progression (GP):

$$
S_n = a \cdot \frac{r^n - 1}{r - 1}, \quad r \neq 1
$$

---

## 🔹 2. Number Theory

* **GCD & LCM**

$$
\text{LCM}(a,b) = \frac{a \cdot b}{\gcd(a,b)}
$$

* **Euler’s Totient**

$$
\phi(n) = n \prod_{p|n} \left(1 - \frac{1}{p}\right)
$$

* **Modular exponentiation**

$$
a^b \pmod{m}
$$

* **Fermat’s Little Theorem**

$$
a^{p-1} \equiv 1 \pmod{p}, \quad p \text{ prime, } a \not\equiv 0
$$

* **Modular inverse**

$$
a^{-1} \equiv a^{p-2} \pmod{p}, \quad p \text{ prime}
$$

---

## 🔹 3. Geometry

* Distance between points:

$$
d = \sqrt{(x_2-x_1)^2 + (y_2-y_1)^2}
$$

* Area of triangle (Heron’s formula):

$$
A = \sqrt{s(s-a)(s-b)(s-c)}, \quad s = \frac{a+b+c}{2}
$$

* Area using coordinates (Shoelace formula):

$$
A = \frac{1}{2} \left| \sum_{i=1}^{n} (x_i y_{i+1} - x_{i+1} y_i) \right|
$$

* Circle area:

$$
A = \pi r^2
$$

---

## 🔹 4. Algebra & Series

* Binomial theorem:

$$
(x+y)^n = \sum_{r=0}^{n} \binom{n}{r} x^{n-r} y^r
$$

* Multinomial theorem:

$$
(x_1 + \dots + x_m)^n = \sum \binom{n}{k_1, \dots, k_m} \prod x_i^{k_i}
$$

* Sum of GP to infinity:

$$
S = \frac{a}{1-r}, \quad |r|<1
$$

---

## 🔹 5. Graph Theory

* Handshake lemma:

$$
\sum_{v \in V} \deg(v) = 2|E|
$$

* Cayley’s formula (number of labeled trees):

$$
n^{n-2}
$$

* Number of spanning trees (Kirchhoff’s theorem) → determinant of Laplacian minor.

---

## 🔹 6. Probability & Statistics

* Conditional probability:

$$
P(A|B) = \frac{P(A \cap B)}{P(B)}
$$

* Bayes theorem:

$$
P(A|B) = \frac{P(B|A) P(A)}{P(B)}
$$

* Expectation linearity:

$$
E[X+Y] = E[X] + E[Y]
$$

* Variance:

$$
Var(X) = E[X^2] - (E[X])^2
$$

---

## 🔹 7. Recurrences & DP

* Fibonacci closed form (Binet’s formula):

$$
F_n = \frac{1}{\sqrt{5}}\left( \left(\frac{1+\sqrt{5}}{2}\right)^n - \left(\frac{1-\sqrt{5}}{2}\right)^n \right)
$$

* Master theorem (divide & conquer):

$$
T(n) = aT\left(\frac{n}{b}\right) + f(n)
$$

---

## 🔹 8. Miscellaneous

* Stirling’s approximation (for large $n!$):

$$
n! \approx \sqrt{2\pi n}\left(\frac{n}{e}\right)^n
$$

* Catalan number:

$$
C_n = \frac{1}{n+1}\binom{2n}{n}
$$

* Harmonic numbers:

$$
H_n = 1 + \frac{1}{2} + \frac{1}{3} + \dots + \frac{1}{n} \approx \ln(n) + \gamma
$$

---

# 🔑 Summary

For contests, the **most useful categories** are:

1. **Series formulas** (fast sum calculation).
2. **Number theory** (GCD, modular arithmetic, primes).
3. **Combinatorics & probability** (nCr, derangements, Catalan).
4. **Geometry formulas** (triangles, polygons, circles).
5. **Asymptotics** (Stirling, harmonic numbers).

---

👉 Do you want me to **assemble all these into a single “Competitive Programming Formula Handbook” (Markdown file)** so you can drop it into your repo and browse like a cheat sheet?
