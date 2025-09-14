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

### Find all Divisors of n
```python
  def divisors(n):
      divs = [1]
      for i in range(2,int(math.sqrt(n))+1):
          if n%i == 0:
              divs.extend([i,n//i])
      divs.extend([n])
      return list(set(divs))
```

### Find GDC of Array
```python
# Euclidean algorithm
      def gcd(a, b):
          while b:
              a, b = b, a % b
          return a

# GCD of array using reduce
    def gcd_a(arr):
      return reduce(gcd, arr)
```

### Count the number of subarrays with **GCD exactly equal to 1**.

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

Perfect 👍 You’re asking about **range queries with GCD** → classic use case for both **Segment Trees** and **Sparse Tables**.
Let me show you both approaches.

---

# 🔹 1. Segment Tree for GCD

### Idea

* Build a segment tree where each node stores the **GCD of its segment**.
* Query `[l, r]` by combining results from relevant segments.
* Update supported (if array changes).

### Python Implementation

```python
from math import gcd

class SegmentTree:
    def __init__(self, arr):
        self.n = len(arr)
        self.tree = [0] * (4 * self.n)  # safe size
        self._build(arr, 0, 0, self.n - 1)
    
    def _build(self, arr, idx, left, right):
        if left == right:
            self.tree[idx] = arr[left]
            return
        mid = (left + right) // 2
        self._build(arr, 2*idx+1, left, mid)
        self._build(arr, 2*idx+2, mid+1, right)
        self.tree[idx] = gcd(self.tree[2*idx+1], self.tree[2*idx+2])
    
    def query(self, ql, qr):
        return self._query(0, 0, self.n - 1, ql, qr)
    
    def _query(self, idx, left, right, ql, qr):
        if ql > right or qr < left:   # disjoint
            return 0
        if ql <= left and right <= qr:  # fully covered
            return self.tree[idx]
        mid = (left + right) // 2
        return gcd(
            self._query(2*idx+1, left, mid, ql, qr),
            self._query(2*idx+2, mid+1, right, ql, qr)
        )

# Example usage
arr = [2, 6, 9, 3, 15]
seg = SegmentTree(arr)
print(seg.query(1, 3))  # gcd(6,9,3) = 3
print(seg.query(0, 4))  # gcd(2,6,9,3,15) = 1
```

**Complexity:**

* Build: `O(n)`
* Query: `O(log n)`
* Update: `O(log n)`

---

# 🔹 2. Sparse Table for GCD

### Idea

* Precompute `st[i][j]` = GCD of subarray starting at `i` of length `2^j`.
* Answer query `[l, r]` by combining **two overlapping intervals** that cover `[l, r]`.
* Immutable array (no updates).

### Python Implementation

```python
from math import gcd, log2

class SparseTable:
    def __init__(self, arr):
        self.n = len(arr)
        self.K = int(log2(self.n)) + 1
        self.st = [[0] * self.K for _ in range(self.n)]
        
        for i in range(self.n):
            self.st[i][0] = arr[i]
        
        j = 1
        while (1 << j) <= self.n:
            i = 0
            while i + (1 << j) <= self.n:
                self.st[i][j] = gcd(self.st[i][j-1], self.st[i + (1 << (j-1))][j-1])
                i += 1
            j += 1
    
    def query(self, l, r):
        j = int(log2(r - l + 1))
        return gcd(self.st[l][j], self.st[r - (1 << j) + 1][j])

# Example usage
arr = [2, 6, 9, 3, 15]
sp = SparseTable(arr)
print(sp.query(1, 3))  # gcd(6,9,3) = 3
print(sp.query(0, 4))  # gcd(2,6,9,3,15) = 1
```

**Complexity:**

* Build: `O(n log n)`
* Query: `O(1)`
* Update: ❌ Not supported efficiently.

---

# ✅ When to use which?

* **Segment Tree** → if array updates are required.
* **Sparse Table** → if array is static and many queries need fast answers.

---

 

