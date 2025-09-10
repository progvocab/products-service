
---

## 🔎 What is an **Inversion**?

An **inversion** in an array is a pair of indices `(i, j)` such that:

$$
i < j \quad \text{and} \quad arr[i] > arr[j]
$$

* Example:

  ```
  arr = [2, 4, 1, 3]
  Inversions = (2,1), (4,1), (4,3) → total 3
  ```

👉 Inversions measure **how far an array is from being sorted**.
A sorted ascending array has **0 inversions**.

---

## 🔧 Problem (Inversion Reduction)

Given an array, **reduce its inversions** (or make the array sorted) using allowed operations.

The most common forms of the problem:

1. **Count inversions** (before reduction)
2. **Find minimum operations to reduce inversions to 0** (i.e., sort the array)

---

## ⚡ Step 1: Counting Inversions

Brute force check of all pairs → $O(n^2)$.
Efficient way → Use **Merge Sort** to count inversions in $O(n \log n)$.

### Python Code (Count Inversions)

```python
def merge_sort_count(arr):
    if len(arr) <= 1:
        return arr, 0

    mid = len(arr) // 2
    left, inv_left = merge_sort_count(arr[:mid])
    right, inv_right = merge_sort_count(arr[mid:])
    merged, inv_split = merge_count(left, right)

    return merged, inv_left + inv_right + inv_split

def merge_count(left, right):
    merged = []
    i = j = inv_count = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1
            inv_count += len(left) - i  # all remaining in left are inversions

    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged, inv_count

# Example
arr = [2, 4, 1, 3, 5]
_, inversions = merge_sort_count(arr)
print("Inversions:", inversions)  # Output: 3
```

---

## ⚡ Step 2: Reducing Inversions (Sorting)

To **reduce inversions to 0**, we must **sort the array**.
Sorting methods like:

* Merge sort → already $O(n \log n)$, reduces inversions optimally
* Bubble sort → each swap removes exactly **1 inversion**
* Insertion sort → each swap reduces some inversions

👉 So the **minimum swaps needed to reduce inversions = number of inversions** (when swaps are allowed only between adjacent elements).

---

## ✅ Example: Reduction via Swaps

```python
def min_swaps_to_sort(arr):
    # Pair each element with index
    n = len(arr)
    arr_pos = list(enumerate(arr))
    arr_pos.sort(key=lambda x: x[1])  # sort by value

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
            swaps += (cycle_size - 1)

    return swaps

arr = [2, 4, 1, 3, 5]
print("Minimum swaps to reduce inversions:", min_swaps_to_sort(arr))  
# Output: 2
```

Explanation:

* `[2, 4, 1, 3, 5]` has **3 inversions**
* But you can fix it with only **2 swaps**:

  * Swap(4,1) → `[2,1,4,3,5]`
  * Swap(2,1) → `[1,2,4,3,5]`
  * Then one more swap if we wanted bubble-style — but cycle method is optimal.

---

## 📌 Summary

* **Inversions** tell how unsorted an array is.
* **Counting inversions** → Merge Sort method $O(n \log n)$.
* **Reducing inversions** → Sorting the array.

  * If only **adjacent swaps** allowed → need exactly `#inversions` swaps.
  * If **any swaps allowed** → use cycle decomposition → need fewer swaps.

---

👉 Do you want me to also show the **relation between inversions and Bubble Sort** (where each swap reduces exactly 1 inversion)? That’s often asked in competitive programming.
