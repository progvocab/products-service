### **Sliding Window Algorithm**

The **sliding window** algorithm is a technique for solving problems involving **contiguous subarrays or substrings**. It maintains a **"window"** over a portion of the data and moves it forward efficiently without recomputing everything from scratch.

---

### **Types of Sliding Window**

1. **Fixed-size window** – window of size `k` slides one step at a time.
2. **Variable-size window** – window expands and shrinks based on conditions.

---

### **Use Cases**

- Maximum sum of a subarray of size `k`
- Longest substring with `k` distinct characters
- Minimum window containing all characters
- Detecting anagrams or duplicates in a window

---

### **Example 1: Fixed-size window – Max sum subarray of size k**

```python
def max_sum_subarray(arr, k):
    n = len(arr)
    window_sum = sum(arr[:k])
    max_sum = window_sum

    for i in range(k, n):
        window_sum += arr[i] - arr[i - k]
        max_sum = max(max_sum, window_sum)

    return max_sum

# Example
arr = [2, 1, 5, 1, 3, 2]
k = 3
print(max_sum_subarray(arr, k))  # Output: 9 (5+1+3)
```

---

### **Example 2: Variable-size window – Longest substring with no repeating characters**

```python
def longest_unique_substring(s):
    char_set = set()
    left = 0
    max_len = 0

    for right in range(len(s)):
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1
        char_set.add(s[right])
        max_len = max(max_len, right - left + 1)

    return max_len

# Example
s = "abcabcbb"
print(longest_unique_substring(s))  # Output: 3 ("abc")
```

---

### **Example 3: Count of subarrays with sum ≤ k**

```python
def count_subarrays_with_sum_lte_k(arr, k):
    left = 0
    total = 0
    count = 0

    for right in range(len(arr)):
        total += arr[right]
        while total > k:
            total -= arr[left]
            left += 1
        count += right - left + 1

    return count

# Example
arr = [1, 2, 3]
k = 4
print(count_subarrays_with_sum_lte_k(arr, k))  # Output: 4
```

---

### **Summary**

| Use Case                                   | Window Type     | Optimized By Sliding Window |
|-------------------------------------------|------------------|------------------------------|
| Max sum in subarray of size `k`           | Fixed            | Avoids recomputing sums     |
| Longest substring with unique chars       | Variable         | Dynamic start/end movement  |
| Anagram detection in string               | Fixed            | Hash maps with sliding scope|
| Subarray sum problems                     | Variable or fixed| Prefix + sliding combo      |

Would you like 2D sliding window examples or how this integrates with prefix sum?