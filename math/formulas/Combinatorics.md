 this is **core math for competitive programming** (permutations, combinations, binomial coefficients, partitions, etc.). Let me give you a **structured cheat sheet** of all important formulas.

---

# 📘 Combinatorics Formulas

## 🔹 Factorials

$$
n! = 1 \cdot 2 \cdot 3 \cdot \dots \cdot n
$$

* Basic building block for permutations & combinations.

---

## 🔹 Permutations

**Number of ways to arrange $r$ objects out of $n$:**

$$
P(n, r) = \frac{n!}{(n-r)!}
$$

* Special case: **arrange all $n$**

  $$
  P(n, n) = n!
  $$

---

## 🔹 Combinations (Binomial Coefficient)

**Number of ways to choose $r$ objects out of $n$ (order doesn’t matter):**

$$
C(n, r) = \binom{n}{r} = \frac{n!}{r!(n-r)!}
$$

* Symmetry property:

  $$
  \binom{n}{r} = \binom{n}{n-r}
  $$

* Pascal’s rule:

  $$
  \binom{n}{r} = \binom{n-1}{r-1} + \binom{n-1}{r}
  $$

---

## 🔹 Variations (Arrangements with Repetition)

* With repetition allowed (each chosen element can be reused):

  $$
  n^r
  $$

---

## 🔹 Combinations with Repetition

(Choosing $r$ items from $n$ types, repetition allowed):

$$
\binom{n+r-1}{r}
$$

---

## 🔹 Multinomial Coefficient

Split $n$ items into groups of sizes $k_1, k_2, \dots, k_m$ (where $\sum k_i = n$):

$$
\binom{n}{k_1, k_2, \dots, k_m} = \frac{n!}{k_1! \, k_2! \, \dots \, k_m!}
$$

---

## 🔹 Derangements (no element in original place)

Number of derangements of $n$ items:

$$
!n = n! \left( \sum_{k=0}^{n} \frac{(-1)^k}{k!} \right)
$$

Approximation:

$$
!n \approx \frac{n!}{e}
$$

---

## 🔹 Binomial Theorem

$$
(x+y)^n = \sum_{r=0}^{n} \binom{n}{r} x^{n-r} y^r
$$

---

## 🔹 Multinomial Theorem

$$
(x_1 + x_2 + \dots + x_m)^n = \sum_{k_1 + k_2 + \dots + k_m = n} \binom{n}{k_1, k_2, \dots, k_m} \prod_{i=1}^m x_i^{k_i}
$$

---

## 🔹 Principle of Inclusion-Exclusion (PIE)

For sets $A_1, A_2, \dots, A_n$:

$$
\Big|\bigcup_{i=1}^n A_i \Big| = \sum |A_i| - \sum |A_i \cap A_j| + \sum |A_i \cap A_j \cap A_k| - \dots
$$

---

## 🔹 Catalan Numbers

Count of valid parentheses sequences, binary trees, etc.

$$
C_n = \frac{1}{n+1} \binom{2n}{n}
$$

---

## 🔹 Stirling Numbers of the Second Kind

Ways to partition $n$ objects into $k$ non-empty subsets:

$$
S(n,k) = \frac{1}{k!} \sum_{j=0}^{k} (-1)^j \binom{k}{j} (k-j)^n
$$

---

## 🔹 Bell Numbers

Number of ways to partition $n$ objects (sum of Stirling numbers):

$$
B_n = \sum_{k=0}^n S(n,k)
$$

---

## 🔹 Principle of Counting (Rule of Product)

If one task can be done in $m$ ways and another in $n$ ways:

$$
\text{Total ways} = m \cdot n
$$

---

# 🔑 Quick Reference Table

| Concept                      | Formula                                                          |
| ---------------------------- | ---------------------------------------------------------------- |
| Permutations                 | $P(n,r) = \frac{n!}{(n-r)!}$                                     |
| Combinations                 | $C(n,r) = \frac{n!}{r!(n-r)!}$                                   |
| Combinations with repetition | $\binom{n+r-1}{r}$                                               |
| Derangements                 | $!n = n! \sum_{k=0}^n \frac{(-1)^k}{k!}$                         |
| Binomial Theorem             | $(x+y)^n = \sum_{r=0}^n \binom{n}{r} x^{n-r} y^r$                |
| Catalan Number               | $C_n = \frac{1}{n+1}\binom{2n}{n}$                               |
| Multinomial                  | $\binom{n}{k_1,\dots,k_m} = \frac{n!}{k_1!\dots k_m!}$           |
| Stirling (2nd kind)          | $S(n,k) = \frac{1}{k!} \sum_{j=0}^k (-1)^j \binom{k}{j} (k-j)^n$ |
| Bell Number                  | $B_n = \sum_{k=0}^n S(n,k)$                                      |

---

⚡ These are the **main combinatorics formulas** used in contests (Codeforces, AtCoder, ICPC).

👉 Do you want me to also include **Python snippets** that compute these efficiently (with factorials, mod prime, nCr mod p, derangements, catalan, etc.) for competitive programming?
