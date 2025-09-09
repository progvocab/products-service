Perfect 👍 Let’s go step by step into the **Binomial Coefficient** — one of the most fundamental topics in combinatorics and competitive programming.

---

# 🔹 What is the Binomial Coefficient?

The **binomial coefficient** $\binom{n}{k}$ (read as “n choose k”) counts the **number of ways to choose $k$ elements from a set of $n$ distinct elements**, without regard to order.

Mathematically:

$$
\binom{n}{k} = \frac{n!}{k!(n-k)!}, \quad 0 \leq k \leq n
$$

---

# 🔹 Properties of Binomial Coefficient

1. **Symmetry**

$$
\binom{n}{k} = \binom{n}{n-k}
$$

👉 Choosing $k$ elements is the same as excluding $n-k$.

---

2. **Pascal’s Identity**

$$
\binom{n}{k} = \binom{n-1}{k-1} + \binom{n-1}{k}
$$

👉 This is the basis of **Pascal’s Triangle**.

---

3. **Boundary Cases**

$$
\binom{n}{0} = \binom{n}{n} = 1
$$

---

4. **Summation Identity**

$$
\sum_{k=0}^n \binom{n}{k} = 2^n
$$

👉 Each element can either be included or excluded → total subsets.

---

5. **Binomial Theorem**

$$
(x+y)^n = \sum_{k=0}^n \binom{n}{k} x^{n-k} y^k
$$

👉 That’s why it’s called *binomial* coefficient.

---

# 🔹 Examples

* $\binom{5}{2} = \frac{5!}{2!3!} = 10$
  (ways to choose 2 elements from 5).

* Expanding $(a+b)^3$:

$$
(a+b)^3 = \binom{3}{0}a^3b^0 + \binom{3}{1}a^2b^1 + \binom{3}{2}ab^2 + \binom{3}{3}b^3
$$

$$
= 1a^3 + 3a^2b + 3ab^2 + 1b^3
$$

---

# 🔹 Python Implementations

### 1. Using `math.comb`

```python
import math

print(math.comb(5, 2))  # 10
```

---

### 2. Manual Factorial Formula

```python
def nCr(n, r):
    if r < 0 or r > n:
        return 0
    fact = lambda x: 1 if x == 0 else x * fact(x-1)
    return fact(n) // (fact(r) * fact(n-r))

print(nCr(5, 2))  # 10
```

---

### 3. Pascal’s Triangle (DP approach)

```python
def build_pascals_triangle(n):
    C = [[0]*(n+1) for _ in range(n+1)]
    for i in range(n+1):
        C[i][0] = C[i][i] = 1
        for j in range(1, i):
            C[i][j] = C[i-1][j-1] + C[i-1][j]
    return C

triangle = build_pascals_triangle(5)
print(triangle[5][2])  # 10
```

---

# 🔹 Competitive Programming Usage

1. **Counting subsets** → $\binom{n}{k}$.
2. **Paths in grid** (right/down moves): $\binom{r+c}{r}$.
3. **Binomial Theorem expansions**.
4. **Probability problems** → combinations in numerator/denominator.
5. **DP with Pascal’s identity** for large $n$.
6. **nCr % MOD** with modular inverses when numbers are huge.

---

👉 Do you want me to show the **efficient method to compute $nCr \pmod{10^9+7}$** (precomputing factorials + modular inverse), since that’s essential for contests?
