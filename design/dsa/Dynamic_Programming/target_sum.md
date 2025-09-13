Here’s a full solution + explanation for **LeetCode 494: Target Sum**.

---

## Problem

You have an integer array `nums` and an integer `target`.
You can put either a `+` or `−` sign in front of each number and then sum them up.
Return *how many different ways* to assign signs so that the resulting sum is `target`.

Example: `nums = [1,1,1,1,1]`, `target = 3` → output = `5`. ([Medium][1])

---

## Key Idea (Transform to Subset Sum)

Let:

* Sum of all numbers in `nums` = `S`.
* Suppose we split `nums` into two groups:

  * One group has the numbers with `+` sign → call its sum `P`.
  * The other group has the numbers with `-` sign → call its sum `N`.

Then:

```
P − N = target       … (1)
P + N = S            … (2)
```

From (1) + (2):

```
2P = target + S  
→ P = (target + S) / 2
```

So the problem reduces to: **how many subsets of `nums` have sum = `(target + S) / 2`**.

But there are constraints:

* `target + S` must be even, because otherwise `P` is not integer.
* `|target|` cannot exceed `S` (if `target > S` or `target < −S`, no solutions).

---

## Solution Approaches

1. **DFS + memoization**: try all `+` and `−` assignments, memoize `(index, current_sum)`. This is exponential in worst case but feasible since `nums.length ≤ 20` and `sum(nums) ≤ 1000`. ([Medium][2])
2. **Subset‐sum DP**: after verifying the conditions, find number of subsets that sum to `P = (target + S)/2`. Use a 1D DP array (or 2D) where `dp[j]` = number of ways to get sum `j`.

---

## Python Code

Here’s a clean DP implementation (1D) that runs in `O(n * sum(nums))` time and `O(sum(nums))` space:

```python
from typing import List

class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        S = sum(nums)
        
        # If it's impossible:
        # 1) target > S or target < -S  
        # 2) (target + S) must be even
        if abs(target) > S:
            return 0
        if (target + S) % 2 != 0:
            return 0
        
        P = (target + S) // 2
        
        # Subset sum: count of subsets with sum = P
        dp = [0] * (P + 1)
        dp[0] = 1
        
        for num in nums:
            # iterate backwards so we don't reuse element multiple times
            for j in range(P, num - 1, -1):
                dp[j] += dp[j - num]
        
        return dp[P]
```

---

## Complexity

* Time: `O(n * P)` where `P = (target + S)/2`, so worst-case `O(n * sum(nums))`.
* Space: `O(P)` for the DP array.

Because constraints are small (`sum(nums) ≤ 1000`, `nums.length ≤ 20`), this is efficient. ([Leetcode][3])

---

If you want, I can also show you the **memoization + DFS version**, or trade‐offs between both. Do you want that?

[1]: https://medium.com/%40cyberseize/leetcode-494-target-sum-56385e070070?utm_source=chatgpt.com "LeetCode 494: Target Sum | by Nikhil Jain"
[2]: https://medium.com/%40natretsel/dsa-stories-leetcode-494-target-sum-2d-dp-f7e0201eb501?utm_source=chatgpt.com "DSA Stories — Leetcode 494. Target Sum (2D DP)"
[3]: https://leetcode.ca/all/494.html?utm_source=chatgpt.com "494. Target Sum - leetcode.ca"
