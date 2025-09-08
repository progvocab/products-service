

# 🔹 Minimum Swaps to Sort (Cycle Decomposition Idea)

When sorting an array, each element can be seen as part of a **permutation cycle**.

👉 Key fact:
A cycle of length `k` can be sorted in **k − 1 swaps**.

---

# 🔎 Why is it `k − 1` and not `k/2`?

Let’s take an example.

### Example: A cycle of length 4

Say the correct order is `[1, 2, 3, 4]` but we have:

```
[2, 3, 4, 1]
```

This is one cycle:

```
1 → 2 → 3 → 4 → 1
```

Length `k = 4`.

---

### Try Sorting It

We want to place every element in its correct position.

1. Swap element at index 0 with where it belongs (put `1` in place):

   ```
   [2, 3, 4, 1] → [1, 3, 4, 2]
   ```
2. Fix the next out-of-place element (`2`):

   ```
   [1, 3, 4, 2] → [1, 2, 4, 3]
   ```
3. Fix the next out-of-place element (`3`):

   ```
   [1, 2, 4, 3] → [1, 2, 3, 4]
   ```

✅ Done in **3 swaps = 4 − 1**.

---

### Why Not `k/2`?

If you think "one swap fixes two elements", that’s **true in some cases (like adjacent swaps)** but not in general cycle sorting.

* Each swap **places one element in its correct position** (sometimes it helps another, but not guaranteed).
* To fix all `k` elements, you’ll need `k − 1` swaps.
* The reason: once `k − 1` elements are in place, the last one automatically falls into place.

So it’s not `k/2`, because you can’t *always* fix two wrong positions in one swap unless the cycle length is exactly `2`.

---

# 🔹 Special Case

* If cycle length `k = 2` → only **1 swap** (which equals `k − 1`, not `k/2` but same numerically).
* For larger cycles, the `k − 1` rule holds.

---

# ✅ Python Code (Minimum Swaps to Sort)

```python
def min_swaps_to_sort(arr):
    n = len(arr)
    arr_pos = sorted(list(enumerate(arr)), key=lambda x: x[1])
    visited = [False] * n
    swaps = 0

    for i in range(n):
        if visited[i] or arr_pos[i][0] == i:
            continue
        
        cycle_size = 0
        j = i
        while not visited[j]:
            visited[j] = True
            j = arr_pos[j][0]
            cycle_size += 1
        
        if cycle_size > 0:
            swaps += cycle_size - 1
    
    return swaps


print(min_swaps_to_sort([4, 3, 2, 1]))   # Output: 2
print(min_swaps_to_sort([2, 3, 4, 1]))   # Output: 3
```

---

# 🔹 Takeaway

* **Cycle of length `k` → needs `k − 1` swaps.**
* Reason: each swap places **at least one element** in its correct position, not necessarily two.
* Only cycles of length `2` "look like" they use 1 swap for 2 elements → but the general formula still holds.

---

👉 Do you want me to also **prove this mathematically** using permutation cycle decomposition (group theory style), not just with examples?
