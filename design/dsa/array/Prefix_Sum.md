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

Great question! Here’s a detailed walkthrough of the **prefix-sum + hashmap** pattern used to count how many **contiguous subarrays** sum to `k`.

---

# The program (for reference)

```python
def subarraySum(nums, k):
    prefix_sum = 0
    count = 0
    seen = {0: 1}
    for num in nums:
        prefix_sum += num
        count += seen.get(prefix_sum - k, 0)
        seen[prefix_sum] = seen.get(prefix_sum, 0) + 1
    return count
```

---

## What each variable means

* **`prefix_sum`**
  Running total up to the current index:
  $\text{prefix\_sum} = \sum_{i=0}^{r} \text{nums}[i]$

* **`seen` (dict: prefix\_sum → frequency)**
  `seen[s]` = how many times a prefix sum equal to `s` has appeared **before** the current index.
  It starts as `{0: 1}` so subarrays that begin at index `0` are counted (more on this below).

* **`count`**
  Total number of contiguous subarrays found so far whose sum is exactly `k`.

---

## Why it works (the math in one line)

For a subarray `nums[l..r]` to sum to `k`:

$$
\text{sum}(l..r) = \text{prefix}[r] - \text{prefix}[l-1] = k
\Rightarrow \text{prefix}[l-1] = \text{prefix}[r] - k
$$

So when we’re at index `r` with current `prefix_sum = prefix[r]`, the number of valid `l` values is exactly **how many times** we have seen a prefix sum of `prefix_sum - k` **earlier**. That’s what `seen[prefix_sum - k]` gives us.

---

## Why `seen = {0: 1}`?

This accounts for subarrays that **start at index 0**.
If `prefix_sum == k` at some `r`, then `prefix_sum - k == 0`. We need one count available for “no elements before start” (i.e., prefix at index `-1`), hence `0:1`.

---

## Why the order of operations matters

Inside the loop we do:

1. `prefix_sum += num`
2. `count += seen.get(prefix_sum - k, 0)`  ← **query previous prefixes first**
3. `seen[prefix_sum] = seen.get(prefix_sum, 0) + 1`  ← **then record current prefix**

If you recorded the current prefix **before** querying, you’d incorrectly allow a **zero-length** subarray at the current position to be counted when `k == 0`.

---

## Dry run (with negatives too)

Example: `nums = [3, 4, 7, 2, -3, 1, 4, 2]`, `k = 7`

Start: `prefix_sum = 0`, `count = 0`, `seen = {0:1}`

| num | prefix\_sum | need = prefix\_sum - k | seen\[need] | count (+=) | update seen\[prefix\_sum] |
| --- | ----------- | ---------------------- | ----------- | ---------- | ------------------------- |
| 3   | 3           | -4                     | 0           | 0          | seen\[3]=1                |
| 4   | 7           | 0                      | 1           | 1          | seen\[7]=1                |
| 7   | 14          | 7                      | 1           | 2          | seen\[14]=1               |
| 2   | 16          | 9                      | 0           | 2          | seen\[16]=1               |
| -3  | 13          | 6                      | 0           | 2          | seen\[13]=1               |
| 1   | 14          | 7                      | 1           | 3          | seen\[14]=2               |
| 4   | 18          | 11                     | 0           | 3          | seen\[18]=1               |
| 2   | 20          | 13                     | 1           | 4          | seen\[20]=1               |

Final `count = 4` subarrays summing to 7:

* `[3, 4]`
* `[7]`
* `[7, 2, -3, 1]`
* `[1, 4, 2]`

---

## Complexity

* **Time:** $O(n)$ — one pass, O(1) average per hashmap op.
* **Space:** $O(n)$ — in worst case all prefix sums are different, so `seen` stores up to `n+1` entries.

---

## Common pitfalls

* Counting **contiguous subarrays** only — this method doesn’t count subsequences.
* Don’t forget `seen = {0:1}` — otherwise you’ll miss subarrays starting at index 0.
* Keep the **order**: query `seen` **before** adding the current `prefix_sum` to it.

---

## Tiny variants

* **Return whether any subarray sums to `k`:** stop early when `seen.get(prefix_sum - k, 0) > 0`.
* **Return indices of one subarray:** store first index of each prefix sum in `seen` instead of frequency.

If you want, I can show the **indices-returning** version too.
