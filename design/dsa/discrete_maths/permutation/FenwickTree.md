 let’s dive into the **Fenwick Tree (Binary Indexed Tree) method** for counting inversions.

This is another standard competitive programming trick, especially useful when the array size is large (up to $10^6$).

---

## 🔹 Idea

1. **Definition Reminder**:
   An inversion is a pair `(i, j)` such that

   $$
   i < j \quad \text{and} \quad A[i] > A[j]
   $$

2. **Observation**:
   If we process the array **from right to left**, for each element `A[i]`,

   * the number of smaller elements already seen (to its right) = number of inversions contributed by `A[i]`.

3. **How to Track?**

   * We need a structure to quickly:

     * Count how many elements ≤ `x` we’ve seen so far.
     * Update frequencies as we move leftward.
   * That’s exactly what a **Fenwick Tree** (BIT) can do in $O(\log n)$.

4. **Coordinate Compression**:
   If values are large (like up to $10^9$), we compress them to ranks (1..n) to fit BIT.

---

## 🔹 Python Code

```python
class FenwickTree:
    def __init__(self, n):
        self.n = n
        self.bit = [0] * (n + 1)

    def update(self, index, delta):
        while index <= self.n:
            self.bit[index] += delta
            index += index & -index

    def query(self, index):
        """Sum from 1 to index"""
        res = 0
        while index > 0:
            res += self.bit[index]
            index -= index & -index
        return res


def count_inversions(arr):
    # Step 1: Coordinate compression
    sorted_unique = sorted(set(arr))
    rank = {val: i+1 for i, val in enumerate(sorted_unique)}  # 1-based

    # Step 2: Init Fenwick Tree
    n = len(sorted_unique)
    bit = FenwickTree(n)

    inversions = 0
    # Step 3: Traverse from right to left
    for num in reversed(arr):
        r = rank[num]
        # How many numbers smaller than current? query(r-1)
        inversions += bit.query(r - 1)
        # Mark current number as seen
        bit.update(r, 1)

    return inversions


# Example
arr = [3, 5, 4, 2, 1]
print("Inversions:", count_inversions(arr))
```

---

## 🔹 Output

```
Inversions: 8
```

---

## 🔹 Step-by-Step Trace (for `[3,5,4,2,1]`)

Process right-to-left:

* Start: BIT empty.
* `1` → query(0) = 0 → inversions = 0, update(1).
* `2` → query(1) = 1 (found \[1]) → inversions = 1, update(2).
* `4` → query(3) = 2 (\[1,2]) → inversions = 3, update(3).
* `5` → query(4) = 3 (\[1,2,4]) → inversions = 6, update(4).
* `3` → query(2) = 2 (\[1,2]) → inversions = 8, update(2).

✅ Done → total inversions = **8**.

---

## 🔹 Complexity

* Building compression: $O(n \log n)$
* For each element: 1 query + 1 update → $O(n \log n)$
* Total: $O(n \log n)$, same as merge sort, but sometimes cleaner for CP.

---

👉 Merge Sort method is more “theoretical” and works directly,
👉 BIT method is very **implementation-friendly** in contests (especially with multiple queries).

---

Do you want me to also show the **Segment Tree method** (generalization of BIT) for inversion counting, or stick with BIT + Merge Sort?
