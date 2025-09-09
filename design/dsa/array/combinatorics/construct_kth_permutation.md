Perfect 👍 This is a **classic competitive programming problem**:

---

# 📖 Problem: K-th Permutation Sequence

You are given:

* `n` → numbers `1..n`
* `k` → an index (1-based)

Find the **k-th permutation** in lexicographic order without generating all permutations.

---

## 🔑 Key Idea

1. **Factorial Number System (Lehmer code)** is used.

   * For `n` numbers, there are `n!` permutations.
   * The first `(n-1)!` permutations start with `1`,
     the next `(n-1)!` with `2`, etc.

2. Process:

   * Compute factorials of numbers up to `n`.
   * Use division & modulo to select elements step by step.
   * Remove the chosen number from the list.

---

## 📌 Example

**n = 4, k = 9**

Permutations in order:

```
1. 1234  
2. 1243  
3. 1324  
4. 1342  
5. 1423  
6. 1432  
7. 2134  
8. 2143  
9. 2314  ✅
```

Steps:

* Numbers = `[1,2,3,4]`
* Total permutations = `4! = 24`
* Each block = `3! = 6`
* k=9 → falls in **second block** (so first digit = `2`)
* Remaining numbers `[1,3,4]`, k=9-6=3
* Next block size = `2! = 2`
* k=3 → 2nd block (digit = `3`)
* Remaining `[1,4]`, k=3-2=1
* Next block size = `1! = 1` → pick `1`, then `4`.
* Result = `2314`. ✅

---

## 🐍 Python Implementation

```python
import math

def kth_permutation(n, k):
    numbers = list(range(1, n+1))
    k -= 1  # convert to 0-based index
    result = []

    for i in range(n, 0, -1):
        fact = math.factorial(i-1)
        index = k // fact
        k %= fact
        result.append(numbers.pop(index))

    return result

# Example
print(kth_permutation(4, 9))  # [2, 3, 1, 4]
print(kth_permutation(3, 5))  # [3, 1, 2]
```

---

## ✅ Output

```
[2, 3, 1, 4]
[3, 1, 2]
```

---

## 🔎 Complexity

* **Time**: `O(n^2)` (because of `pop` from list each step)
* **Optimized**: Use balanced tree (Fenwick/Segment tree) for `O(n log n)` removal.
* **Space**: `O(n)`.

---

## 🚀 Common CP Variants

1. **Return the k-th permutation as a string (LeetCode 60)**
2. **Find index (rank) of a given permutation**
3. **K-th permutation with constraints** (e.g., circular permutations, fixed positions)

---

👉 Do you want me to also show the **reverse problem**: given a permutation, find its **rank (k)** in lexicographic order?
