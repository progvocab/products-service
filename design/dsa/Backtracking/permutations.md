Great! Let's walk through **a simple problem using backtracking** for both **combinations** and **permutations**, similar in spirit to what we just did earlier.

---

## ✅ Problem 1: **Generate all combinations of k numbers out of 1 to n**

### ✨ Example:

Input: `n = 4`, `k = 2`
Output:

```
[ [1,2], [1,3], [1,4], [2,3], [2,4], [3,4] ]
```

### 🔁 Backtracking Logic:

* At each step, choose a number and move forward.
* Stop when length `k` is reached.
* Use pruning to skip numbers already used or out of bounds.

### 🧠 Time Complexity:

Roughly `O(C(n, k))` – number of combinations.

### ✅ Code in Python:

```python
def combine(n, k):
    result = []

    def backtrack(start, path):
        if len(path) == k:
            result.append(path[:])
            return
        for i in range(start, n + 1):
            path.append(i)
            backtrack(i + 1, path)
            path.pop()  # backtrack

    backtrack(1, [])
    return result

print(combine(4, 2))
```

---

## ✅ Problem 2: **Generate all permutations of a list**

### ✨ Example:

Input: `nums = [1, 2, 3]`
Output:

```
[ [1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1] ]
```

### 🔁 Backtracking Logic:

* At each step, pick a number that hasn't been used yet.
* Track used numbers via a boolean array or in-place swapping.

### 🧠 Time Complexity:

`O(n!)` – number of permutations.

### ✅ Code in Python:

```python
def permute(nums):
    result = []
    used = [False] * len(nums)

    def backtrack(path):
        if len(path) == len(nums):
            result.append(path[:])
            return
        for i in range(len(nums)):
            if not used[i]:
                used[i] = True
                path.append(nums[i])
                backtrack(path)
                path.pop()
                used[i] = False

    backtrack([])
    return result

print(permute([1, 2, 3]))
```

---

Would you like to explore these with constraints (like duplicate handling, sum limits, etc.) next?
