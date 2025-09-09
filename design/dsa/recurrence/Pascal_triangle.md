

Let’s analyze **Pascal’s Triangle** recurrence.

---

## 🔹 Pascal’s Triangle Recurrence

Each entry is defined as:

$$
C(n, k) = C(n-1, k-1) + C(n-1, k)
$$

with base cases:

$$
C(n, 0) = C(n, n) = 1
$$

---

## 🔎 Order of Recurrence

* To compute $C(n, k)$, we use **two different previous values**:

  * one from the **previous row, previous column** ($C(n-1, k-1)$)
  * one from the **previous row, same column** ($C(n-1, k)$)

👉 So, this is **not first-order** (depends on more than one previous value).

It’s also **not exactly second-order in the classical "single index" sense** (like Fibonacci depends only on $n-1, n-2$).

Instead, it is a **two-dimensional recurrence relation**.

* First-order in **row index** $n$ (since only $n-1$ row is used).
* But **two-term dependency** in column index $k$.

---

## 📝 Intuition

* **Sum of array**: $S(n) = S(n-1) + n$ → **first-order (1 previous term)**
* **Fibonacci**: $F(n) = F(n-1) + F(n-2)$ → **second-order (2 previous terms)**
* **Pascal’s triangle**: $C(n, k) = C(n-1,k-1) + C(n-1,k)$ → **two-dimensional recurrence**, not strictly first/second-order.

---

## 🐍 Python Implementation

```python
def pascal_triangle(n):
    triangle = [[0]*(n+1) for _ in range(n+1)]
    for i in range(n+1):
        triangle[i][0] = triangle[i][i] = 1
        for j in range(1, i):
            triangle[i][j] = triangle[i-1][j-1] + triangle[i-1][j]
    return triangle

# Example: print first 5 rows
rows = pascal_triangle(5)
for r in rows:
    print([x for x in r if x != 0])
```

✅ Output:

```
[1]
[1, 1]
[1, 2, 1]
[1, 3, 3, 1]
[1, 4, 6, 4, 1]
[1, 5, 10, 10, 5, 1]
```

---

✨ So: **Pascal’s Triangle is a 2D recurrence relation, not classified as first-order or second-order in the 1D sense.**

👉 Do you want me to show how to solve binomial coefficient recurrence using **combinatorics formula** instead of DP?

Great 🚀 Let’s solve the **Pascal’s Triangle recurrence** using **combinatorics (closed form)** instead of DP.

---

## 🔹 Recurrence Reminder

$$
C(n, k) = C(n-1, k-1) + C(n-1, k), \quad C(n,0)=C(n,n)=1
$$

---

## 🔹 Closed-Form Solution (Binomial Coefficient)

Using combinatorics, we know:

$$
C(n, k) = \binom{n}{k} = \frac{n!}{k!(n-k)!}
$$

This closed form directly gives the value without building Pascal’s triangle row by row.

---

## 🔹 Why Does It Work?

Interpretation:

* $\binom{n}{k}$ = number of ways to choose $k$ objects from $n$.
* Splitting on whether the first object is chosen or not:

  * If chosen → choose remaining $k-1$ from $n-1$: $\binom{n-1}{k-1}$
  * If not chosen → choose $k$ from $n-1$: $\binom{n-1}{k}$

This directly leads to the recurrence:

$$
\binom{n}{k} = \binom{n-1}{k-1} + \binom{n-1}{k}
$$

---

## 🐍 Python (Closed Form)

```python
import math

def nCr(n, r):
    if r < 0 or r > n:
        return 0
    return math.comb(n, r)  # Python 3.8+ has built-in comb()

# Example: print first 6 rows of Pascal's Triangle
for n in range(6):
    row = [nCr(n, k) for k in range(n+1)]
    print(row)
```

✅ Output:

```
[1]
[1, 1]
[1, 2, 1]
[1, 3, 3, 1]
[1, 4, 6, 4, 1]
[1, 5, 10, 10, 5, 1]
```

---

## 🔹 Competitive Programming Insight

* **DP version (recurrence)** is useful when $n$ is large and we want *all coefficients* up to $n$.
* **Closed form** ($\binom{n}{k}$) is better if we need a *single coefficient*, especially with modular arithmetic.

For **modulo $M$** (like $10^9+7$), we compute:

$$
\binom{n}{k} \equiv \frac{n!}{k!(n-k)!} \pmod{M}
$$

using **modular inverse (Fermat’s Little Theorem)**.

---

👉 Do you want me to also show how to compute **nCr under modulo** efficiently (precomputing factorials + modular inverse), since that’s a very common competitive programming trick?
