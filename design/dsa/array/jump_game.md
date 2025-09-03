Great choice 👌 — **Jump Game** is a very popular array/greedy problem. Let’s break it down step by step:

---

# 🔹 Problem Statement (LeetCode 55 — Jump Game)

You are given an array `nums` where each element represents the **maximum jump length** at that position.

👉 Task: Determine if you can reach the **last index** starting from the first index.

---

### Example 1

```python
nums = [2,3,1,1,4]
```

* Start at index `0` → jump up to `2` steps.
* Jump to index `1` (value = 3) → can reach last index.
  ✅ Output: `True`

---

### Example 2

```python
nums = [3,2,1,0,4]
```

* Index `0` → can go max to index `3`.
* At index `3`, jump length = `0` → stuck!
* Cannot reach index `4`.
  ❌ Output: `False`

---

# 🔹 Approaches

### 1. **Greedy Approach (Optimal O(n))**

* Keep track of the **furthest index you can reach**.
* Iterate over each index:

  * If the current index is beyond the furthest reachable, return `False`.
  * Update `furthest = max(furthest, i + nums[i])`.
* If we finish loop, return `True`.

✅ Time: `O(n)`
✅ Space: `O(1)`

---

### Python Code (Greedy)

```python
def canJump(nums):
    furthest = 0
    for i, jump in enumerate(nums):
        if i > furthest:
            return False
        furthest = max(furthest, i + jump)
    return True
```

---

### 2. **Dynamic Programming (O(n²)) – Slower**

* For each index, check if any previous index can jump to it.
* Very slow for large arrays.

```python
def canJumpDP(nums):
    n = len(nums)
    dp = [False] * n
    dp[0] = True

    for i in range(1, n):
        for j in range(i):
            if dp[j] and j + nums[j] >= i:
                dp[i] = True
                break
    return dp[-1]
```

---

# 🔹 Comparison

| Approach                | Time Complexity | Space Complexity | Notes                     |
| ----------------------- | --------------- | ---------------- | ------------------------- |
| **Greedy**              | O(n)            | O(1)             | Best, efficient           |
| **Dynamic Programming** | O(n²)           | O(n)             | Too slow for large inputs |

---

✅ **Summary**:

* Jump Game asks: *Can we reach the last index?*
* The **greedy algorithm** is the optimal solution: keep track of the furthest index you can reach.

---

👉 Do you want me to also explain the harder version — **Jump Game II** (minimum jumps required)?
