

In **competitive programming**, recurrence relations often appear in **DP transitions** or **counting problems**.
Let’s focus on **first-order recurrence relations** and how to solve them efficiently.

---

# 📖 1. What is a First-Order Recurrence?

A **first-order recurrence relation** expresses $T(n)$ in terms of **only the previous term** $T(n-1)$:

$$
T(n) = a \cdot T(n-1) + f(n), \quad \text{with initial condition } T(0) \text{ or } T(1).
$$

Where:

* $a$ = constant multiplier,
* $f(n)$ = extra function (could be constant, polynomial, exponential, etc.).

---

# 🔑 2. General Solution Method

### Case 1: **Homogeneous recurrence**

$$
T(n) = a \cdot T(n-1)
$$

* Solution:

$$
T(n) = T(0) \cdot a^n
$$

---

### Case 2: **Non-homogeneous recurrence**

$$
T(n) = a \cdot T(n-1) + b
$$

(“affine recurrence”, very common in DP)

* Expand:

$$
T(n) = a^n \cdot T(0) + b \cdot \frac{a^n - 1}{a - 1}, \quad \text{if } a \neq 1
$$

* If $a = 1$:

$$
T(n) = T(0) + n \cdot b
$$

---

### Case 3: With polynomial/exponential $f(n)$

If $T(n) = a \cdot T(n-1) + f(n)$,
use **iteration / unrolling** until you spot a pattern.
For example:

$$
T(n) = T(n-1) + n
$$

Unroll:

$$
T(n) = T(0) + 1 + 2 + \dots + n = T(0) + \frac{n(n+1)}{2}
$$

---

# 📌 3. Examples in Competitive Programming

### Example 1: Compound Interest

$$
A(n) = r \cdot A(n-1), \quad A(0) = P
$$

Solution:

$$
A(n) = P \cdot r^n
$$

---

### Example 2: Arithmetic Sum Recurrence

$$
S(n) = S(n-1) + n, \quad S(0)=0
$$

Solution:

$$
S(n) = \frac{n(n+1)}{2}
$$

---

### Example 3: DP with Constant Addition

$$
T(n) = 2T(n-1) + 3, \quad T(0)=1
$$

Solution:

$$
T(n) = 2^n \cdot T(0) + 3 \cdot (2^n - 1)
$$

$$
T(n) = 2^n + 3(2^n - 1) = 4\cdot 2^n - 3
$$

---

# 🐍 Python Implementation

```python
def solve_first_order(a, b, T0, n):
    """Solves T(n) = a*T(n-1) + b, with T(0) = T0"""
    if a == 1:
        return T0 + n * b
    return (a**n) * T0 + b * (a**n - 1) // (a - 1)


# Example: T(n) = 2T(n-1) + 3, T(0)=1
for n in range(6):
    print(f"T({n}) =", solve_first_order(2, 3, 1, n))
```

✅ Output:

```
T(0) = 1
T(1) = 5
T(2) = 13
T(3) = 29
T(4) = 61
T(5) = 125
```

---

# 🚀 How This Helps in CP

* **DP optimization**: Find closed-form instead of recomputing.
* **Math-based problems** (like sums, investments, population growth).
* **Modulo arithmetic**: Often results are required mod $10^9+7$, so use **modular exponentiation** for $a^n$.

---

👉 Do you want me to also show how to handle **second-order recurrences** (like Fibonacci), which are even more common in competitive programming?

---

Let’s carefully check whether the **sum of first $n$ natural numbers** is a **first-order recurrence**.

---

## ✅ Sum of Array / Natural Numbers

Define:

$$
S(n) = 1 + 2 + 3 + \dots + n
$$

We can express it as:

$$
S(n) = S(n-1) + n, \quad S(0)=0
$$

This matches the **form of a first-order recurrence**:

$$
T(n) = a \cdot T(n-1) + f(n)
$$

where $a = 1$, $f(n) = n$.

---

## 🔎 So Yes:

The **sum of the first $n$ numbers** is indeed a **first-order recurrence**, because it depends **only on the immediately previous term**.

---

## 🎯 Solution

Unrolling gives:

$$
S(n) = S(0) + 1 + 2 + \dots + n
$$

$$
S(n) = \frac{n(n+1)}{2}
$$

---

## 🐍 Python Example

```python
def sum_first_n(n):
    # Recurrence: S(n) = S(n-1) + n
    if n == 0:
        return 0
    return sum_first_n(n-1) + n

# Closed-form formula
def sum_formula(n):
    return n * (n + 1) // 2

print([sum_first_n(i) for i in range(6)])
print([sum_formula(i) for i in range(6)])
```

✅ Output:

```
[0, 1, 3, 6, 10, 15]
[0, 1, 3, 6, 10, 15]
```

---

👉 Would you like me to also show **why this is NOT second-order** (like Fibonacci) and contrast the two?
