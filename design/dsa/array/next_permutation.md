Got it 👍 You’re asking about the **Next Permutation** problem.

This is a classic algorithm problem (LeetCode #31). Let’s go step by step.

---

## 🔹 Problem Statement

Given a list of numbers, rearrange them into the **next lexicographically greater permutation**.
If no such permutation exists (i.e., the array is sorted in descending order), rearrange into the **lowest possible order** (ascending).

👉 Example:

* Input:  `[1, 2, 3]` → Output: `[1, 3, 2]`
* Input:  `[3, 2, 1]` → Output: `[1, 2, 3]`
* Input:  `[1, 1, 5]` → Output: `[1, 5, 1]`

---

## 🔹 Algorithm (Steps)

1. Traverse from **right to left** and find the first index `i` such that `nums[i] < nums[i+1]`.
   (This is the "pivot" — the place where order can be improved.)
2. If no such index exists → array is the **last permutation**, just reverse it.
3. Otherwise:

   * From right to left, find the first index `j` such that `nums[j] > nums[i]`.
   * Swap `nums[i]` and `nums[j]`.
   * Reverse the part from `i+1` to end (to make it the smallest possible).

---

## 🔹 Python Implementation

```python
def nextPermutation(nums):
    n = len(nums)
    
    # Step 1: Find first decreasing element from the right
    i = n - 2
    while i >= 0 and nums[i] >= nums[i + 1]:
        i -= 1
    
    if i >= 0:  # If not fully descending
        # Step 2: Find element just larger than nums[i]
        j = n - 1
        while nums[j] <= nums[i]:
            j -= 1
        # Swap
        nums[i], nums[j] = nums[j], nums[i]
    
    # Step 3: Reverse the suffix
    left, right = i + 1, n - 1
    while left < right:
        nums[left], nums[right] = nums[right], nums[left]
        left += 1
        right -= 1
    
    return nums


# Example usage
nums = [1, 2, 3]
print("Next permutation:", nextPermutation(nums))  # [1, 3, 2]

nums = [3, 2, 1]
print("Next permutation:", nextPermutation(nums))  # [1, 2, 3]

nums = [1, 1, 5]
print("Next permutation:", nextPermutation(nums))  # [1, 5, 1]
```

---

## 🔹 Complexity

* **Time Complexity**: `O(n)` (single traversal and reversal).
* **Space Complexity**: `O(1)` (in-place).

---

Would you like me to also extend this so it **generates all permutations in order** (like iterating through each next permutation until we return to the start)?
