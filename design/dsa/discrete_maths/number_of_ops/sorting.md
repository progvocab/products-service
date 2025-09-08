

# 🔹 Minimum Swaps to Sort (Cycle Decomposition Idea)

- When sorting an array, each element can be seen as part of a **permutation cycle**.

- Every permutation can be uniquely expressed as a product of disjoint cycles meaning every element moves within its own independent cycle. 
- To find the disjoint cycle decomposition of a permutation, start with an element, trace its path through the permutation until it returns to its starting position, and write this as a cycle.
- Repeat this process with any remaining elements that haven't been included in a cycle until all elements are accounted for. 


the permutation (1 3 2 5)(4 1 3): 
1. Start with 1:
From (4 1 3), 1 goes to 4.
From (1 3 2 5), 4 goes to 3.
From (4 1 3), 3 goes to nothing in this step.
From (1 3 2 5), 3 goes to 2.
From (1 3 2 5), 2 goes to nothing in this step.
From (4 1 3), 2 goes to nothing in this step.
From (1 3 2 5), 2 goes to 1.
So, we have the cycle (1 4 3 2).
2. Next element: 5:
From (1 3 2 5), 5 goes to 1. But 1 is already in our cycle.
So, the element 5 is a fixed point and can be written as the cycle (5).
Therefore, the disjoint cycle decomposition is (1 4 3 2)(5). 

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


This is the **core beauty** of the minimum-swaps-to-sort algorithm 🙂. Let’s carefully unpack **why just visiting unvisited elements (and counting cycles) is enough** to compute the number of swaps.

---

# 🔹 Step 1: Sorting = Rearranging a Permutation

When you compare an array with its sorted version, you can think of it as a **permutation of indices**.

Example:

```
arr = [4, 3, 2, 1]
sorted = [1, 2, 3, 4]
```

If we track original indices:

```
arr_pos = [(3,1), (2,2), (1,3), (0,4)]
```

This tells us:

* Position 0 should get the element from index 3
* Position 1 should get the element from index 2
* Position 2 should get the element from index 1
* Position 3 should get the element from index 0

That’s a **permutation of indices**:

```
0 → 3 → 0,   1 → 2 → 1
```

---

# 🔹 Step 2: Permutations Decompose into Cycles

Every permutation can be split into **disjoint cycles**.
Here, we have 2 cycles:

* Cycle 1: (0, 3)
* Cycle 2: (1, 2)

---

# 🔹 Step 3: Sorting a Cycle

Inside a cycle of length `k`:

* Every element is in the wrong place
* To sort it, you must rotate elements until all are correct
* This always takes exactly `k − 1` swaps

### Example

Cycle = (0, 3), elements `[4, 1]`

Swaps:

```
[4, 3, 2, 1]
swap → [1, 3, 2, 4] ✅ cycle fixed in 1 = (2−1) swap
```

Cycle = (1, 2), elements `[3, 2]`

```
[1, 3, 2, 4]
swap → [1, 2, 3, 4] ✅ cycle fixed in 1 = (2−1) swap
```

Total = 2 swaps.

---

# 🔹 Step 4: Why Just Visiting Works

When you “visit” unvisited elements:

* You are traversing a cycle
* Once you’ve seen the whole cycle, you know its length = `cycle_size`
* Minimum swaps for that cycle = `cycle_size − 1`

So the algorithm doesn’t need to *actually perform* the swaps — it only needs to count them using cycle sizes.

---

# 🔹 General Formula

If the permutation has `c` disjoint cycles with lengths `k₁, k₂, …, k_c`, then:

```
Minimum swaps = Σ (kᵢ − 1) = (n − c)
```

(where `n` = number of elements, `c` = number of cycles).

---

# ✅ Visualization

For `arr = [4,3,2,1]`:

* n = 4
* cycles = 2
* answer = 4 − 2 = 2 swaps ✔

---

# 🔹 Intuition

* Each cycle is like a “closed loop” of misplaced elements.
* Sorting the cycle requires putting everyone in their place, which takes `k − 1` moves.
* By visiting non-visited elements, we’re essentially counting the cycle sizes → and from that, the number of swaps.

---

👉 Do you want me to **show the same with a graph diagram** (nodes = indices, arrows = where element should go), so you can *see the cycles* and why it’s `k − 1`?


Perfect 👍 now let’s tackle **minimum rotations to sort an array**.

---

# 🔹 Problem: Minimum Rotations to Sort an Array

**Operation allowed**: rotate array left or right (cyclic shift).
**Goal**: find the minimum number of rotations needed to make the array sorted.
If not possible, return `-1`.

---

## 🔹 Step 1: When is it possible?

A rotated array is just a **cyclic shift** of a sorted array.

Example:
Sorted array = `[1,2,3,4,5]`
Rotations:

```
[1,2,3,4,5]   (0 shifts)
[2,3,4,5,1]   (1 shift)
[3,4,5,1,2]   (2 shifts)
[4,5,1,2,3]   (3 shifts)
[5,1,2,3,4]   (4 shifts)
```

👉 So an array is sortable by rotation **iff it is a rotation of the sorted version**.

---

## 🔹 Step 2: How to check?

* Let `target = sorted(arr)`
* Make a double array: `target + target`
* Check if `arr` appears inside.
* Index = number of rotations needed.

---

## 🔹 Step 3: Python Implementation

```python
def min_rotations_to_sort(arr):
    n = len(arr)
    target = sorted(arr)

    # Concatenate sorted array with itself
    doubled = target + target

    # Try to match arr inside doubled
    for i in range(n):
        if doubled[i:i+n] == arr:
            return i  # min rotations
    return -1  # not possible


# 🔹 Examples
print(min_rotations_to_sort([3,4,5,1,2]))  # 3 rotations
print(min_rotations_to_sort([2,3,4,5,1]))  # 1 rotation
print(min_rotations_to_sort([1,2,3,4,5]))  # 0 rotations
print(min_rotations_to_sort([2,1,3,4,5]))  # -1 (not a rotation of sorted)
```

---

## 🔎 Walkthrough

Example: `[3,4,5,1,2]`

* Sorted target = `[1,2,3,4,5]`
* Doubled = `[1,2,3,4,5,1,2,3,4,5]`
* Check windows of size 5:

  * Found match at index 2 → need **2 rotations left** (or 3 right).

---

## 🔹 Complexity

* Naive check = O(n²).
* Can be improved with **KMP (string matching)** → O(n).

---

✅ Summary:

* **Swaps** → use cycle decomposition.
* **Reversals** → pancake greedy.
* **Rotations** → check if array is a rotation of sorted version (string matching trick).

---

👉 Do you want me to also cover **“minimum adjacent swaps to sort”** (where only neighbors can be swapped)? That’s another common competitive programming variant.


