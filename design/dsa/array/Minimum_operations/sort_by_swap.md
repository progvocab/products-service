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

---

Another Example 

---


This is a **minimum swaps to sort the array** problem, which can be solved using **cycle decomposition**. Let’s solve it step by step for `[5, 3, 7, 2]`.

---

### Step 1: Pair each element with its index

We want to know where each element should go after sorting.

Original array:

```
Index: 0  1  2  3
Value: 5  3  7  2
```

Sorted array: `[2, 3, 5, 7]`

So we pair value with original index:

```
[(5,0), (3,1), (7,2), (2,3)]
```

After sorting by values:

```
[(2,3), (3,1), (5,0), (7,2)]
```

---

### Step 2: Detect cycles

Now, map **current index → correct index**.

* At index `0`, value is `5` (original index 0).
  In sorted array, `5` should be at index `2`.
  So `0 → 2`.

* At index `2`, value is `7` (original index 2).
  In sorted array, `7` should be at index `3`.
  So `2 → 3`.

* At index `3`, value is `2` (original index 3).
  In sorted array, `2` should be at index `0`.
  So `3 → 0`.

That makes cycle: **0 → 2 → 3 → 0** (length 3).

* At index `1`, value `3` is already in the right place (cycle length 1).

---

### Step 3: Formula

For a cycle of length `k`, swaps needed = `k - 1`.

So:

* Cycle \[0,2,3] → length = 3 → needs `2` swaps.
* Cycle \[1] → length = 1 → needs `0` swaps.

---

✅ **Answer = 2 swaps**

---

### Python Code

```python
def min_swaps_to_sort(arr):
    n = len(arr)
    # Pair each element with its index
    arrpos = sorted([(val, i) for i, val in enumerate(arr)])
    visited = [False] * n
    swaps = 0

    for i in range(n):
        # Already visited or already in correct place
        if visited[i] or arrpos[i][1] == i:
            continue

        # Find cycle length
        cycle_size = 0
        j = i
        while not visited[j]:
            visited[j] = True
            j = arrpos[j][1]
            cycle_size += 1

        if cycle_size > 1:
            swaps += cycle_size - 1

    return swaps

print(min_swaps_to_sort([5, 3, 7, 2]))  # Output: 2
```

---

Would you like me to also show how this same problem can be solved using **graph cycle decomposition visualization** (drawing arrows of permutation)?
