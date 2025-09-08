Let’s carefully check whether the permutation

```
[4, 3, 2, 1]
```

is **even or odd**.

---

# 🔹 Step 1: Write it as a permutation of indices

We assume the permutation is on `1..4` (not `0..3`).
So it maps:

```
1 → 4  
2 → 3  
3 → 2  
4 → 1
```

---

# 🔹 Step 2: Cycle Decomposition

Let’s trace cycles:

* Start at 1: `1 → 4 → 1` → cycle `(1 4)`
* Next unvisited = 2: `2 → 3 → 2` → cycle `(2 3)`

So the permutation is:

```
(1 4)(2 3)
```

---

# 🔹 Step 3: Count Transpositions

* Cycle `(1 4)` = 1 swap
* Cycle `(2 3)` = 1 swap
* Total = 2 swaps = **even**

So the permutation is **even** ✅

---

# 🔹 Step 4: Verify with Inversions

Count pairs `(i,j)` where `i<j` and `A[i]>A[j]`.

Array = `[4, 3, 2, 1]`

* 4 > 3,2,1 → 3 inversions
* 3 > 2,1 → 2 inversions
* 2 > 1 → 1 inversion
* 1 → 0 inversions

Total = `3+2+1 = 6` inversions → even → **even permutation** ✅

---

✅ Final Answer:
The permutation `[4,3,2,1]` is **even**.

---

👉 Do you want me to also show how to compute parity for **any array in Python quickly** (O(n log n) using merge sort)? That’s useful in competitive programming when arrays are large.


Perfect 👍 Let’s write a clean **Python function** that checks if a permutation is **even or odd**.
We’ll use the **inversion count method** (works in both 0-based and 1-based arrays).

---

## ✅ Python Code

```python
def permutation_parity(perm):
    """
    Return 'even' or 'odd' depending on the permutation parity.
    Works for 0-based or 1-based permutations.
    """
    n = len(perm)
    inversions = 0
    
    # Count inversions
    for i in range(n):
        for j in range(i+1, n):
            if perm[i] > perm[j]:
                inversions += 1

    return "even" if inversions % 2 == 0 else "odd"


# 🔹 Examples
print(permutation_parity([4,3,2,1]))   # even (6 inversions)
print(permutation_parity([1,2,3,4]))   # even (0 inversions)
print(permutation_parity([2,1,3,4]))   # odd  (1 inversion)
print(permutation_parity([3,1,2]))     # even (2 inversions)
```

---

## 🔎 Output

```
even
even
odd
even
```

---

⚡ Note:

* `[4,3,2,1]` → 6 inversions → even
* `[2,1,3,4]` → 1 inversion → odd

---

👉 Do you want me to also give you an **O(n log n)** version (using merge sort) so it works for very large permutations in competitive programming?
