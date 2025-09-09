 **count inversions using Merge Sort** — the classic $O(n \log n)$ method that shows up in competitive programming.

We’ll use your example array `[3,5,4,2,1]`.

---

# 🔹 Idea

When merging two sorted halves:

* If `left[i] <= right[j]`, no inversion (just copy `left[i]`).
* If `left[i] > right[j]`, then **all remaining elements in `left[i:]` are also > right\[j]**
  → so we add `len(left) - i` inversions at once.

This lets us count inversions efficiently while merging.

---

# 🔹 Python Code

```python
def merge_count_split_inv(left, right):
    i = j = inv_count = 0
    merged = []
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            # all remaining elements in left[i:] are greater than right[j]
            inv_count += len(left) - i
            j += 1
    
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged, inv_count


def sort_count(arr):
    n = len(arr)
    if n <= 1:
        return arr, 0
    
    mid = n // 2
    left, inv_left = sort_count(arr[:mid])
    right, inv_right = sort_count(arr[mid:])
    merged, inv_split = merge_count_split_inv(left, right)
    
    return merged, inv_left + inv_right + inv_split


# Example
arr = [3, 5, 4, 2, 1]
sorted_arr, inversions = sort_count(arr)
print("Sorted:", sorted_arr)
print("Inversions:", inversions)
```

---

# 🔹 Step-by-Step Trace for `[3,5,4,2,1]`

1. Split into `[3,5]` and `[4,2,1]`.
2. Left `[3,5]`: 0 inversions.
3. Right `[4,2,1]`:

   * Split into `[4]` and `[2,1]`.
   * `[2,1]` → merge finds 1 inversion (`(2,1)`), sorted → `[1,2]`.
   * Merge `[4]` and `[1,2]`:

     * `4 > 1` → add 1 inversion.
     * `4 > 2` → add 1 inversion.
       → 2 inversions.
   * Total right = 3 inversions.
4. Merge `[3,5]` and `[1,2,4]`:

   * `3 > 1` → add 2 (because `[3,5]` both > 1).
   * `3 > 2` → add 2 (because `[3,5]` both > 2).
   * `5 > 4` → add 1.
     → 5 inversions.

👉 Total inversions = **0 + 3 + 5 = 8**. ✅

---

# 🔹 Output

```
Sorted: [1, 2, 3, 4, 5]
Inversions: 8
```

---

# 🔹 Why This Is Useful

* Runs in $O(n \log n)$, much faster than $O(n^2)$ brute force.
* Works on big arrays (10^5 or more).
* Frequently asked in CP contests as:

  * “Count number of inversions”
  * “Minimum adjacent swaps to sort”

---

Would you like me to also show the **Fenwick Tree (BIT) method** for inversion counting? That’s the other standard CP trick (especially useful if elements are huge, after coordinate compression).
