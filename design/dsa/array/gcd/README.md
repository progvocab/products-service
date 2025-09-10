**competitive programming problem related to GCD of subarrays**. Let me give you a structured breakdown:

---

## Typical Problem Statements Related to GCD of Subarrays

1. **Find maximum GCD among all subarrays**

   * Given an array `A[1…n]`, find the maximum possible GCD of any subarray.

2. **Count number of subarrays with GCD = k**

   * Given array `A[1…n]` and integer `k`, count how many subarrays have GCD exactly equal to `k`.

3. **Longest subarray with GCD > 1**

   * Find the maximum length of a contiguous subarray where the GCD of all elements is greater than 1.

4. **Queries on subarray GCD**

   * Given `q` queries of form `(l, r)`, compute `gcd(A[l..r])`.

---

## Key Observations and Tricks

* **Property of GCD in subarrays**:
  If you expand a subarray by adding elements, its GCD can only **stay the same or decrease**.
  This means:

  * You don’t need to check all subarrays independently.
  * You can build results incrementally.

* **Efficient Trick (Dynamic Map of GCDs)**
  For every ending index `i`, keep a map/set of `{gcd_value : count_of_subarrays}` that end at `i`.

  * Transition: For each `(g, cnt)` from previous index, new gcd = `gcd(g, A[i])`.
  * Add also the single element `A[i]`.
  * This way, you don’t re-check all subarrays, you just update gcd states.

* **Segment Tree / Sparse Table**
  For **queries** (find GCD of `[l, r]`), build a sparse table or segment tree with GCD function.

  * Preprocessing: `O(n log n)`
  * Query: `O(1)` with Sparse Table, `O(log n)` with Segment Tree.

---

## Example Problem and Python Solution

### Problem:

Count the number of subarrays with **GCD exactly equal to 1**.

### Solution Idea:

* Iterate array.
* Maintain dictionary of gcd values for subarrays ending at current position.
* Merge with current element.
* Count subarrays where gcd = 1.

```python
from math import gcd
from collections import defaultdict

def count_subarrays_gcd_one(arr):
    n = len(arr)
    gcd_map = defaultdict(int)
    total = 0
    
    for x in arr:
        new_map = defaultdict(int)
        # Single element subarray
        new_map[x] += 1
        
        # Extend previous subarrays
        for g, cnt in gcd_map.items():
            new_g = gcd(g, x)
            new_map[new_g] += cnt
        
        gcd_map = new_map
        
        total += gcd_map[1]  # Count subarrays with GCD=1 at this step
    
    return total

# Example
arr = [2, 3, 4, 6]
print(count_subarrays_gcd_one(arr))  # Output -> number of subarrays with gcd=1
```

---

⚡ This same approach generalizes to:

* Counting subarrays with `gcd = k` (just count `gcd_map[k]` each step).
* Finding maximum gcd of any subarray (track max key in `gcd_map`).
* Finding longest subarray with gcd > 1 (track lengths).

---

👉 Do you want me to **pick one concrete competitive programming problem** (say from Codeforces/LeetCode/Hackerrank) and explain its solution in full, or should I give you a **general template** for solving *any subarray GCD* problem?
