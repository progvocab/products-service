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
