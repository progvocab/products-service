this is a common point of confusion in competitive programming.

Let’s carefully break it down.

---

## 🔹 Cycle Decomposition Method for Minimum Swaps to Sort

When we want to **sort an array with minimum swaps**, we analyze the permutation formed by the array’s indices.

* The array can be represented as a **permutation** of indices.
* That permutation can be decomposed into **cycles**.
* Each cycle of length $k$ requires exactly **$k - 1$ swaps** to fix.

---

## 🔹 Why $k - 1$?

Suppose you have a cycle of length $k$.

* To place every element in its correct position, you can repeatedly swap one misplaced element with the one that should be there.
* After $k-1$ swaps, the cycle is resolved.

**Example:**
Array = `[4, 3, 2, 1]` → permutation cycles: `(1 4)(2 3)`

* Each cycle has length 2 → needs `2 - 1 = 1` swap.
* Total swaps = 2. ✅


---
 Let’s implement the **minimum swaps to sort** using **cycle decomposition** in Python.

---

## 🔹 Python Implementation

```python
def min_swaps_to_sort(arr):
    n = len(arr)

    # Pair each element with its index and sort by value
    arrpos = [(val, idx) for idx, val in enumerate(arr)]
    arrpos.sort(key=lambda x: x[0])

    visited = [False] * n
    swaps = 0

    for i in range(n):
        # if element is visited or already in correct position
        if visited[i] or arrpos[i][1] == i:
            continue

        # compute the size of this cycle
        cycle_size = 0
        j = i
        while not visited[j]:
            visited[j] = True
            j = arrpos[j][1]  # move to next index
            cycle_size += 1

        # if cycle has k nodes, it takes k-1 swaps
        if cycle_size > 0:
            swaps += (cycle_size - 1)

    return swaps
```

---

## 🔹 Example Runs

```python
print(min_swaps_to_sort([4, 3, 2, 1]))   # Output: 2
print(min_swaps_to_sort([1, 5, 4, 3, 2])) # Output: 2
print(min_swaps_to_sort([2, 3, 4, 1, 5])) # Output: 3
```

---

## 🔹 Explanation of Example `[4, 3, 2, 1]`

* Sorted: `[1, 2, 3, 4]`
* Permutation cycles: `(1 4)(2 3)`
* Two cycles of length 2 → needs `1 + 1 = 2` swaps. ✅

---

👉 Do you also want me to extend this code to **return the actual swap operations** (not just the count), so you can see which elements get swapped?
