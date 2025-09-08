  "minimum number of operations" is one of the most **common problem types** in competitive programming.

Different problems have different kinds of "operations" allowed (swap, reverse, rotate, increment, etc.), so the approach depends on **what operation is defined**. Let’s break it down systematically:

---

# 🔹 1. Minimum Swaps to Sort an Array

* **Operation**: swap any two elements.
* **Approach**: Use **cycle decomposition**.
* **Reason**: Each cycle of length `k` requires `k-1` swaps to fix.
* **Formula**:

  $$
  \text{min_swaps} = \sum (k_i - 1) = n - \text{number_of_cycles}
  $$

✅ Code snippet:

```python
def min_swaps_to_sort(arr):
    n = len(arr)
    sorted_arr = sorted([(val, i) for i, val in enumerate(arr)])
    visited = [False] * n
    swaps = 0

    for i in range(n):
        if visited[i] or sorted_arr[i][1] == i:
            continue
        cycle_size, j = 0, i
        while not visited[j]:
            visited[j] = True
            j = sorted_arr[j][1]
            cycle_size += 1
        if cycle_size > 1:
            swaps += cycle_size - 1
    return swaps
```

---

# 🔹 2. Minimum Reversals

* **Operation**: reverse a prefix or subarray.
* **Examples**:

  * Pancake sorting → reverse prefixes until sorted.
  * Sorting signed brackets → use greedy counting.
* **Approach**: Often greedy or simulate cycle structure with reversals.

---

# 🔹 3. Minimum Rotations

* **Operation**: rotate left/right.
* **Approach**: Use string matching tricks.

  * To rotate string `A` into `B`, check if `B` is in `A+A`.
  * Index of first occurrence gives min rotations.

---

# 🔹 4. Minimum Increments/Decrements

* **Operation**: add or subtract 1 from elements.
* **Approach**: Use math / greedy.

  * E.g., to make all elements equal → move toward median.

---

# 🔹 5. General Strategy

👉 To solve *any* "minimum ops" problem:

1. **Identify invariant** (something that doesn’t change after an operation).

   * Example: parity of permutation is invariant under swaps of adjacent pairs.
2. **Decompose into cycles / components**.

   * Sorting often reduces to counting cycles.
3. **Greedy vs DP vs Graph**:

   * If each operation has a *cost* → treat as a graph problem (BFS/shortest path).
   * If multiple ways to reduce problem → DP.
   * If operations are structured (like swaps, rotations) → greedy + cycle decomposition.

---

# 🔹 Example: Minimum Swaps (Why `k-1`?)

Cycle of length `k`:

* To fix it, you can "place elements one by one" in correct spot.
* Each swap fixes **one element in place**.
* After `k-1` swaps, all `k` are fixed.
  👉 That’s why not `k/2`.

---

# 🔎 Summary Cheat Sheet

| Operation type      | Typical solution                         |
| ------------------- | ---------------------------------------- |
| Swap any elements   | Cycle decomposition (`n - cycles`)       |
| Swap adjacent only  | Count inversions (merge sort O(n log n)) |
| Rotate string       | Substring trick with `A+A`               |
| Reverse prefix      | Pancake sorting (greedy, O(n²))          |
| Increment/decrement | Median or DP                             |
| Complex ops w/cost  | BFS or shortest path                     |

---

✅ So the trick is: **find the invariant** (cycle, inversion, sum, gcd, etc.) → then minimize operations needed to reach the target state.

---

Do you want me to take **one specific operation type** (say, swap + rotate or swap + reverse) and walk you through how to derive the **minimum number of operations formula** step by step, with Python code?
