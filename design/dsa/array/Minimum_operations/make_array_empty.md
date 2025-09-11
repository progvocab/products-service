this is **LeetCode 2870 – Minimum Number of Operations to Make Array Empty**.

---

## 📌 Problem Statement

You are given an integer array `nums`.

* In one operation, you may **remove exactly 2 or 3 elements that are equal** from the array.
* Return the **minimum number of operations** to make the array empty.
* If it’s not possible, return `-1`.

---

## 🔑 Key Insight

* We only care about the **frequency** of each number.
* For each frequency `f`, we want to split it into groups of **2** or **3**.
* The **minimum operations** = sum of minimum operations for each frequency.

### Rules:

* If `f == 1` → impossible → return `-1`.
* Otherwise, always try to use as many **3s** as possible, since they reduce operation count.
* Special case: if `f % 3 == 1`, we need to replace one group of `3+1` with `2+2`.

---

## 🛠️ Strategy

For each frequency `f`:

* If `f % 3 == 0` → use all 3s → `f // 3` ops.
* If `f % 3 == 1` → need `(f-4)//3` 3s + 2 twos → `(f // 3 - 1) + 2 = f // 3 + 1`.
* If `f % 3 == 2` → use `(f // 3)` 3s + one 2 → `f // 3 + 1`.

So in all cases (except `f=1`):
👉 Minimum operations = `ceil(f / 3)`.

---

## ✅ Python Solution

```python
from collections import Counter
import math

def minOperations(nums):
    freq = Counter(nums)
    ans = 0
    
    for f in freq.values():
        if f == 1:
            return -1
        ans += math.ceil(f / 3)
    
    return ans
```

---

## 🔎 Example Walkthroughs

### Example 1

```
nums = [2,3,3,2,2,4,2,3,4]
freq = {2:4, 3:3, 4:2}

- For 2 → f=4 → ceil(4/3)=2 ops
- For 3 → f=3 → ceil(3/3)=1 op
- For 4 → f=2 → ceil(2/3)=1 op

Total = 2+1+1 = 4
```

✅ Output = 4

---

### Example 2

```
nums = [2,1,2,2,3,3]
freq = {2:3, 1:1, 3:2}

- For 2 → f=3 → 1 op
- For 1 → f=1 → impossible → return -1
```

✅ Output = -1

---

## ⏱️ Complexity

* Counting: `O(n)`
* Processing: `O(n)` (at most `n` unique values)
* **Overall:** `O(n)` time, `O(n)` space

---

Do you want me to also show you why **`ceil(f/3)` always works** (with a proof using `f mod 3` cases)?
