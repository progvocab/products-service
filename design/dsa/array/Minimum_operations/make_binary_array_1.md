Here’s a full explanation + solution for **LeetCode 3191 — Minimum Operations to Make Binary Array Elements Equal to One I**:

---

## Problem Statement

You are given a binary array `nums` (containing only 0s and 1s). You can apply the following operation any number of times:

* Choose **any 3 consecutive elements** in the array, and **flip** all of them (flip means: 0 → 1, 1 → 0).

Return the **minimum number of operations** required to make *all* elements in `nums` equal to 1. If it is impossible, return `-1`.

### Examples

* Example 1:
  Input: `nums = [0,1,1,1,0,0]`
  Output: `3`
  One way:

  1. Flip indices 0,1,2 → array becomes `[1,0,0,1,0,0]`
  2. Flip indices 1,2,3 → `[1,1,1,0,0,0]`
  3. Flip indices 3,4,5 → `[1,1,1,1,1,1]`

* Example 2:
  Input: `nums = [0,1,1,1]`
  Output: `-1`
  Because you cannot flip the last 0 in a 3-window once you get near the end.

---

## Key Observations / Reasoning

1. **Greedy approach** works here:
   You scan from left to right. Whenever you see a `0` at position `i`, you *must* flip starting at `i`, because that's the only way to convert that `0` to `1` since any flip needs a window of length 3. If you don’t flip at `i` when you see a `0`, that `0` will never get flipped (since any future window that includes `i` must start at `i-1`, `i-2`, etc., but we've already passed those if we’re scanning left to right).

2. Flipping a window starting at `i` affects `nums[i]`, `nums[i+1]`, `nums[i+2]`. After this flip, `nums[i]` becomes `1`.

3. If you reach a position `i` such that `i + 2 >= len(nums)` and `nums[i] == 0`, you can’t perform a valid flip window starting at `i` (there aren’t three elements). So it’s impossible → return `-1`.

4. After processing all possible windows, if you manage to convert all elements to `1`, then the total number of flips you did is the minimum.

---

## Solution in Python

```python
class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        operations = 0
        
        for i in range(n - 2):  # up to index n-3 inclusive
            if nums[i] == 0:
                # flip nums[i], nums[i+1], nums[i+2]
                nums[i] ^= 1
                nums[i+1] ^= 1
                nums[i+2] ^= 1
                operations += 1
        
        # after doing all possible flips, check last two positions
        if nums[-1] == 0 or nums[-2] == 0:
            return -1
        
        return operations
```

---

## Complexity

* **Time Complexity**: $O(n)$, because we do one pass over the array, flipping in constant time when needed.
* **Space Complexity**: $O(1)$ extra space (we modify the array in place, aside from counters and indices).

---

If you like, I can also show you a version that *doesn’t modify* the input array (using a copy or simulated flips) or discuss what makes some arrays impossible in more depth. Do you want that?
