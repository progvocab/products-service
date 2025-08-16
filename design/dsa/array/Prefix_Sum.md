### **Prefix Sum Algorithm (Cumulative Sum)**

The **prefix sum** algorithm computes a new array where each element at index `i` is the sum of all elements from index `0` to `i` in the original array.

---

### **Basic Idea**
Given an array `arr = [a0, a1, a2, ..., an]`, the prefix sum array `prefix` is:

```text
prefix[0] = arr[0]  
prefix[1] = arr[0] + arr[1]  
prefix[2] = arr[0] + arr[1] + arr[2]  
...  
prefix[i] = prefix[i-1] + arr[i]
```

---

### **Python Code Example**

```python
def prefix_sum(arr):
    prefix = [0] * len(arr)
    prefix[0] = arr[0]
    for i in range(1, len(arr)):
        prefix[i] = prefix[i - 1] + arr[i]
    return prefix

# Example
arr = [2, 4, 6, 8]
print(prefix_sum(arr))  # Output: [2, 6, 12, 20]
```

---

### **Use Cases**

1. **Efficient Range Sum Queries**  
   Instead of summing a subarray every time (O(n)), use prefix sum for O(1) range sum:
   ```python
   range_sum = prefix[j] - prefix[i - 1]  # sum from i to j
   ```

2. **Subarray Sum Problems**  
   e.g., find number of subarrays with given sum.

3. **2D Prefix Sum (Cumulative Matrix)**  
   Used in image processing and dynamic programming on grids.

4. **Difference Array Technique**  
   For applying range updates in constant time.

---

### **Range Sum Query Example**

```python
def build_prefix_sum(arr):
    prefix = [0] * (len(arr) + 1)
    for i in range(len(arr)):
        prefix[i + 1] = prefix[i] + arr[i]
    return prefix

def range_sum(prefix, left, right):
    return prefix[right + 1] - prefix[left]

arr = [2, 4, 6, 8, 10]
prefix = build_prefix_sum(arr)
print(range_sum(prefix, 1, 3))  # Output: 18 (4 + 6 + 8)
```

---

Would you like examples for 2D prefix sum or use in competitive problems?