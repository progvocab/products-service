Great! Let’s dive into the **Pigeonhole Principle** 🕊️

---

# 🔹 Pigeonhole Principle — Definition

The **Pigeonhole Principle (PHP)** says:

> If you put $n$ items into $m$ containers (pigeonholes), and if $n > m$, then at least one container must contain **more than one item**.

---

# 🔹 Intuition

* Imagine $13$ socks placed into $12$ drawers.
  ✅ At least one drawer has ≥ 2 socks.

* More generally, if $n$ objects are placed into $m$ boxes,
  then **one box contains at least ⌈n/m⌉ objects.**

---

# 🔹 Applications in Competitive Programming

1. **Duplicates must exist**

   * Given $n+1$ integers from $1 \dots n$, there must be at least one duplicate.
2. **Birthday paradox**

   * With 367 people, at least 2 share a birthday (pigeonholes = 366 days).
3. **Remainder argument**

   * Among any `n+1` integers, at least two have the same remainder mod `n`.

---

# 🔹 Example Problem

**Problem:** Given an array of size $n+1$ where each element is in range \[1…n], find any duplicate element.
This is guaranteed by the pigeonhole principle.

---

# 🔹 Python Implementation

```python
def find_duplicate(arr):
    """
    Pigeonhole principle: 
    arr has length n+1, elements in [1..n]
    guaranteed at least one duplicate
    """
    seen = set()
    for num in arr:
        if num in seen:
            return num
        seen.add(num)
    return None  # should never happen due to PHP
```

---

# 🔹 Example Run

```python
arr = [1, 3, 4, 2, 2]
print("Duplicate:", find_duplicate(arr))  # Output: 2

arr = [5, 4, 3, 2, 1, 5]
print("Duplicate:", find_duplicate(arr))  # Output: 5
```

---

# 🔹 Another Example (mod pigeonhole)

**Problem:** Given any 5 integers, prove that at least two have the same remainder mod 4.

```python
def pigeon_mod(nums, m):
    remainders = {}
    for num in nums:
        r = num % m
        if r in remainders:
            return (remainders[r], num)  # found collision
        remainders[r] = num
    return None

nums = [7, 13, 18, 22, 26]  # 5 numbers, 4 pigeonholes
print("Collision mod 4:", pigeon_mod(nums, 4))
```

✅ Output (example):

```
Collision mod 4: (18, 22)  # both ≡ 2 mod 4
```

---

# 🔹 Key Takeaway

* Pigeonhole Principle **guarantees existence**, but often doesn’t tell us **which pigeonhole**.
* In programming, we combine PHP with **hashing, sets, modulo arithmetic** to actually **find the duplicates or collisions**.

---

Would you like me to show you **classic competitive programming problems based on PHP** (like birthday paradox, subset sums, etc.) with code?
