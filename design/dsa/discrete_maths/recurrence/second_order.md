 let’s contrast **first-order vs second-order recurrences** with examples from competitive programming.

---

# 🔹 1. First-Order Recurrence

👉 **Definition:**
A recurrence is **first-order** if the value of $T(n)$ depends **only on the previous term** $T(n-1)$.

### Example: **Sum of first $n$ numbers**

$$
S(n) = S(n-1) + n, \quad S(0)=0
$$

* Depends only on $S(n-1)$.
* This is **first-order**.

📌 Closed form:

$$
S(n) = \frac{n(n+1)}{2}
$$

✅ Python:

```python
def sum_first_order(n):
    if n == 0:
        return 0
    return sum_first_order(n-1) + n
```

---

# 🔹 2. Second-Order Recurrence

👉 **Definition:**
A recurrence is **second-order** if the value of $T(n)$ depends on the **two previous terms** $T(n-1)$ and $T(n-2)$.

### Example: **Fibonacci Sequence**

$$
F(n) = F(n-1) + F(n-2), \quad F(0)=0, F(1)=1
$$

* Depends on **two previous states**.
* This is **second-order**.

📌 Closed form (Binet’s Formula):

$$
F(n) = \frac{\varphi^n - (1-\varphi)^n}{\sqrt{5}}, \quad \varphi=\frac{1+\sqrt{5}}{2}
$$

✅ Python:

```python
def fib_second_order(n):
    if n <= 1:
        return n
    return fib_second_order(n-1) + fib_second_order(n-2)
```

---

# 🔎 Key Differences

| Type             | General Form                      | Example                  | Complexity                               |
| ---------------- | --------------------------------- | ------------------------ | ---------------------------------------- |
| **First-Order**  | $T(n) = aT(n-1) + f(n)$           | Sum of first $n$ numbers | Often solvable with direct formula       |
| **Second-Order** | $T(n) = aT(n-1) + bT(n-2) + f(n)$ | Fibonacci sequence       | Usually needs matrix exponentiation / DP |

---

# 🔥 Competitive Programming Insight

* **First-order recurrences** often simplify to arithmetic or geometric progressions.
* **Second-order recurrences** usually need **DP or matrix exponentiation** for fast computation ($O(\log n)$).

---

👉 Do you want me to also cover **general method to solve linear recurrences** (characteristic equation method), which is very useful in contests for problems like Fibonacci, Tribonacci, etc.?
