Perfect 👍 — let’s connect the **number theory formulas** I explained with **actual LeetCode problems** where they’re useful.

---

## 🔑 Number Theory Formulas → LeetCode Problems

### 1. **GCD / LCM**

* Formula:

  $$
  \text{lcm}(a, b) = \frac{a \cdot b}{\gcd(a, b)}
  $$
* 🔹 Problems:

  * **[LeetCode 1979 – Find Greatest Common Divisor of Array](https://leetcode.com/problems/find-greatest-common-divisor-of-array/)**
  * **[LeetCode 914 – X of a Kind in a Deck of Cards](https://leetcode.com/problems/x-of-a-kind-in-a-deck-of-cards/)** → needs GCD across frequencies.
  * **[LeetCode 1447 – Simplified Fractions](https://leetcode.com/problems/simplified-fractions/)** → reduce fractions using gcd.

---

### 2. **Prime Numbers / Sieve**

* Formula: Sieve of Eratosthenes in $O(n \log \log n)$.
* 🔹 Problems:

  * **[LeetCode 204 – Count Primes](https://leetcode.com/problems/count-primes/)** → sieve.
  * **[LeetCode 1175 – Prime Arrangements](https://leetcode.com/problems/prime-arrangements/)** → factorial arrangements split between primes and non-primes.
  * **[LeetCode 866 – Prime Palindrome](https://leetcode.com/problems/prime-palindrome/)**.

---

### 3. **Divisors**

* Formula:

  
  $d(n) = (e_1+1)(e_2+1)\dots$
 
* 🔹 Problems:

  * **[LeetCode 1390 – Four Divisors](https://leetcode.com/problems/four-divisors/)** → divisor counting.
  * **[LeetCode 2521 – Distinct Prime Factors of Product of Array](https://leetcode.com/problems/distinct-prime-factors-of-product-of-array/)**.

---

### 4. **Modular Arithmetic & Fast Power**

* Formula:

  $(a+b) \bmod m, \quad (a \cdot b) \bmod m$
  
* Binary exponentiation:

  $$a^b \bmod m \quad \text{in } O(\log b)$$
* 🔹 Problems:

  * **[LeetCode 372 – Super Pow](https://leetcode.com/problems/super-pow/)** → fast exponentiation.
  * **[LeetCode 50 – Pow(x, n)](https://leetcode.com/problems/powx-n/)** → binary exponentiation.
  * **[LeetCode 1922 – Count Good Numbers](https://leetcode.com/problems/count-good-numbers/)** → mod exp.

---

### 5. **Euler’s Totient (φ)**

* Formula:

  $$\phi(n) = n \prod_{p|n}\left(1 - \frac{1}{p}\right)$$
* 🔹 Problems:

  * **[LeetCode 2654 – Minimum Number of Operations to Make All Array Elements Equal to 1](https://leetcode.com/problems/minimum-number-of-operations-to-make-all-array-elements-equal-to-1/)** → relies on gcd coprime properties.
  * **[LeetCode 972 – Equal Rational Numbers](https://leetcode.com/problems/equal-rational-numbers/)** → reduces with coprimes.

---

### 6. **Chinese Remainder Theorem (CRT)**

* Formula: Solve system of congruences if moduli are coprime.
* 🔹 Problems:

  * **[LeetCode 1187 – Make Array Strictly Increasing](https://leetcode.com/problems/make-array-strictly-increasing/)** (not direct CRT, but modular constraints appear).
  * **[LeetCode 466 – Count the Repetitions](https://leetcode.com/problems/count-the-repetitions/)** → requires modular cycle reasoning.
  * Many custom contest problems use CRT when syncing cycles (like bus schedule problems).

---

### 7. **Fermat’s Little Theorem**

* Formula:

  $$a^{p-1} \equiv 1 \pmod{p}$$

  Modular inverse:

  $$a^{-1} \equiv a^{p-2} \pmod{p}$$
* 🔹 Problems:

  * **[LeetCode 1862 – Sum of Floored Pairs](https://leetcode.com/problems/sum-of-floored-pairs/)** (modular arithmetic for large sums).
  * **[LeetCode 1735 – Count Ways to Make Array With Product](https://leetcode.com/problems/count-ways-to-make-array-with-product/)** → uses nCr % p with modular inverse.
  * **[LeetCode 1006 – Clumsy Factorial](https://leetcode.com/problems/clumsy-factorial/)** → mod properties.

---

### 8. **Diophantine Equations**

* Formula: $ax + by = c$, solvable if gcd(a, b) divides c.
* 🔹 Problems:

  * **[LeetCode 365 – Water and Jug Problem](https://leetcode.com/problems/water-and-jug-problem/)** → classic linear Diophantine equation.
  * **[LeetCode 879 – Profitable Schemes](https://leetcode.com/problems/profitable-schemes/)** → constraints with gcd.

---

### 9. **Combinatorics + Modular Inverse**

* 🔹 Problems:

  * **[LeetCode 62 – Unique Paths](https://leetcode.com/problems/unique-paths/)** → binomial coefficients.
  * **[LeetCode 1643 – Kth Smallest Instructions](https://leetcode.com/problems/kth-smallest-instructions/)** → combinatorics with nCr.
  * **\[LeetCode 1000+ variations in contests using nCr % 1e9+7**].

---

✅ So, **direct number theory problems** on LeetCode include:

* 1979, 914, 1447 (GCD/LCM)
* 204, 1175 (Primes)
* 1390, 2521 (Divisors)
* 372, 1922 (Fast Power)
* 2654, 365 (Totient, Diophantine)
* 1735 (Fermat’s theorem + modular inverse)

---

👉 Do you want me to prepare a **mapping table** (formula → problem links) in markdown so you can use it as a quick contest reference?
