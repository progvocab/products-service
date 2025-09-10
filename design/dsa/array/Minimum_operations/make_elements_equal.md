 about the **“Make All Elements Equal”** type of problem (common in Codeforces, LeetCode, GFG, etc.).

Let me explain the **general problem statement, solution approach, formulas, and Python code**.

---

# 📘 Problem: Make All Elements Equal

You are given an array `arr` of size `n`.
In one operation, you can **change an element’s value** (depends on the exact problem version: increment/decrement/replace).
The goal is to find the **minimum number of operations** to make all elements equal.

---

## 🔹 Version 1: Replace Any Element with Any Value

* If you are allowed to **replace an element with any value**, the best target is the **most frequent element**.
* Because every other element must change to that value.
* So the formula is:

$$
\text{min operations} = n - \text{(frequency of most common element)}
$$

---

### ✅ Example 1

```
arr = [3, 3, 2, 1, 3]
n = 5
Most frequent = 3 (appears 3 times)
Answer = 5 - 3 = 2
```

So, 2 operations are enough (change 2 and 1 → to 3).

---

## 🔹 Version 2: Only Increment/Decrement Allowed

* If you can only **increment or decrement** elements, the best target is the **median** of the array.
* Why? Because the median minimizes the **sum of absolute differences**:

$$
\text{min operations} = \sum_{i=1}^n |a_i - \text{median}(a)|
$$

---

### ✅ Example 2

```
arr = [1, 2, 3]
Median = 2
Answer = |1-2| + |2-2| + |3-2| = 1 + 0 + 1 = 2
```

---

## 🔹 Version 3: Only Increment Allowed (GCD Method)

* If you can **only increase** elements, the optimal target is the **maximum element**.
* Formula:

$$
\text{min operations} = \sum_{i=1}^n (\max(a) - a_i)
$$

---

### ✅ Example 3

```
arr = [2, 4, 6]
Max = 6
Answer = (6-2) + (6-4) + (6-6) = 4 + 2 + 0 = 6
```

---

# 🔹 Python Solutions

```python
from collections import Counter

# Version 1: Replace with any value
def min_ops_replace_any(arr):
    freq = Counter(arr)
    most_common = max(freq.values())
    return len(arr) - most_common


# Version 2: Increment/Decrement allowed (Median method)
def min_ops_median(arr):
    arr.sort()
    n = len(arr)
    median = arr[n // 2]
    return sum(abs(x - median) for x in arr)


# Version 3: Only Increment allowed
def min_ops_increment(arr):
    max_val = max(arr)
    return sum(max_val - x for x in arr)
```

---

## 🔹 Test

```python
print(min_ops_replace_any([3, 3, 2, 1, 3]))  # 2
print(min_ops_median([1, 2, 3]))             # 2
print(min_ops_increment([2, 4, 6]))          # 6
```

---

✅ So depending on the **problem version** (replace any value / increment-decrement / only increment), the formula differs.

---

👉 Do you want me to also prepare a **table of “Make All Elements Equal” variants** (with operation allowed, formula, and complexity) for quick CP reference?
