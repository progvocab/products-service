

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
