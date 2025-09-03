Great question! Let’s go step by step and fully understand the **Longest Increasing Subsequence (LIS)** problem and how to solve it using **Dynamic Programming** in Python.

---

## 🔹 Problem Definition

Given an array of integers, find the length of the **longest increasing subsequence** (LIS).

* **Subsequence** means elements that appear in the same order, but not necessarily contiguous.
* **Increasing** means each element in the subsequence must be strictly greater than the previous one.

👉 Example:
Input: `nums = [10, 9, 2, 5, 3, 7, 101, 18]`
Output: `4`
Explanation: The LIS is `[2, 3, 7, 101]` (length = 4).

---

## 🔹 Dynamic Programming Approach

We use a **1D DP array** where:

* `dp[i]` = the length of the **longest increasing subsequence ending at index `i`**.

### Steps:

1. Initialize `dp = [1] * n` (each element itself is a subsequence of length 1).
2. For each index `i` (from left to right):

   * Look at all indices `j < i`.
   * If `nums[j] < nums[i]`, it means we can append `nums[i]` after LIS ending at `j`.
   * Update:

     ```
     dp[i] = max(dp[i], dp[j] + 1)
     ```
3. Answer = `max(dp)` (the longest LIS anywhere in the array).

---

## 🔹 Python Code

```python
def lengthOfLIS(nums):
    n = len(nums)
    if n == 0:
        return 0

    dp = [1] * n  # Each number is an LIS of length 1

    for i in range(n):
        for j in range(i):
            if nums[j] < nums[i]:  # Increasing condition
                dp[i] = max(dp[i], dp[j] + 1)

    return max(dp)


# Example usage
nums = [10, 9, 2, 5, 3, 7, 101, 18]
print("Length of LIS:", lengthOfLIS(nums))
```

---

## 🔹 Complexity

* **Time Complexity**: `O(n^2)` (two nested loops).
* **Space Complexity**: `O(n)` (the `dp` array).

---

## 🔹 Optimized Approach (Binary Search)

There is an **O(n log n)** solution using **Patience Sorting + Binary Search**:

* Maintain an array `sub` that stores the smallest possible tail of increasing subsequences.
* Use `bisect_left` to replace elements or append.

👉 That version is faster for large inputs.

---

Would you like me to also show you the **O(n log n)** optimized binary search solution in Python, alongside the DP one?
