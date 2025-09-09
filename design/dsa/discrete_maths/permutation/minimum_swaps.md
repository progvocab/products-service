Awesome — here’s the clean connection between **minimum swaps**, **cycle decomposition**, and **permutation parity**.

# How they relate

* Write the array as a permutation of indices mapping **current position → sorted position**.
* Decompose this permutation into **disjoint cycles**.
* If the permutation has `c` cycles on `n` elements, then

$$
\textbf{min\_swaps} \;=\; \sum (k_i-1) \;=\; n - c.
$$

Why? Each cycle of length $k$ needs exactly $k-1$ swaps to fix (each swap “locks in” at least one element; after $k-1$ swaps the last is forced into place).

## Parity connection

* A cycle of length $k$ has parity $(k-1)\bmod 2$ (because it can be written as $k-1$ transpositions).
* The whole permutation’s parity is the sum over cycles:

  $$
  \text{parity} \equiv \sum (k_i-1) \equiv n - c \pmod{2}.
  $$
* But $n-c$ is exactly **min\_swaps**.
  **Therefore:**

  $$
  \text{(minimum swaps)} \bmod 2 \;=\; \text{(permutation parity)}.
  $$

  * Even permutation ⇒ even minimum number of swaps.
  * Odd permutation ⇒ odd minimum number of swaps.

---

## Python: compute min swaps, cycles, and check parity two ways

```python
def min_swaps_to_sort(arr):
    n = len(arr)
    arr_pos = sorted([(val, i) for i, val in enumerate(arr)], key=lambda x: x[0])
    visited = [False] * n
    swaps = 0
    cycles = 0

    for i in range(n):
        if visited[i]:
            continue
        # walk a new cycle
        cycle_len = 0
        j = i
        while not visited[j]:
            visited[j] = True
            j = arr_pos[j][1]
            cycle_len += 1
        if cycle_len > 0:
            cycles += 1
            if cycle_len > 1:
                swaps += cycle_len - 1

    # n - cycles equals swaps (may include 1-cycles implicitly)
    assert swaps == n - cycles
    return swaps, cycles

def permutation_parity_by_cycles(arr):
    # parity = (n - cycles) % 2
    swaps, cycles = min_swaps_to_sort(arr[:])
    n = len(arr)
    return "even" if (n - cycles) % 2 == 0 else "odd"

def permutation_parity_by_inversions(arr):
    inv = 0
    n = len(arr)
    for i in range(n):
        for j in range(i+1, n):
            if arr[i] > arr[j]:
                inv += 1
    return "even" if inv % 2 == 0 else "odd"

# Examples
tests = [
    [4,3,2,1],     # two 2-cycles → min swaps 2 → even
    [2,3,4,1,5],   # one 4-cycle + fixed → min swaps 3 → odd
    [1,5,4,3,2],   # cycle structure → min swaps 2 → even
]

for a in tests:
    swaps, cycles = min_swaps_to_sort(a[:])
    p1 = permutation_parity_by_cycles(a[:])
    p2 = permutation_parity_by_inversions(a[:])
    print(f"{a} -> min_swaps={swaps}, cycles={cycles}, parity(cycles)={p1}, parity(inversions)={p2}")
```

### What you’ll see

* `min_swaps = n - cycles`.
* `parity(cycles)` equals `parity(inversions)`, and both match `min_swaps % 2`.

If you’d like, I can also give you an **O(n log n)** inversion counter (Fenwick/Merge Sort) to handle big arrays efficiently.


Great choice 👍 Let’s carefully go through the array

$$
A = [3, 5, 4, 2, 1]
$$

and explain everything step by step.

---

## 🔹 Step 1: Sorted version

Sorted array = `[1, 2, 3, 4, 5]`.

---

## 🔹 Step 2: Index mapping (permutation representation)

We want to see: **where does each element currently sit, vs. where should it go in the sorted array?**

* Element `1` → should be at index `0`, but is at index `4`.
* Element `2` → should be at index `1`, but is at index `3`.
* Element `3` → should be at index `2`, but is at index `0`.
* Element `4` → should be at index `3`, but is at index `2`.
* Element `5` → should be at index `4`, but is at index `1`.

So mapping **current index → correct index**:

```
0 → 2   (A[0]=3 should go to index 2)
1 → 4   (A[1]=5 should go to index 4)
2 → 3   (A[2]=4 should go to index 3)
3 → 1   (A[3]=2 should go to index 1)
4 → 0   (A[4]=1 should go to index 0)
```

Permutation (on indices):

$$
P = [2, 4, 3, 1, 0]
$$

---

## 🔹 Step 3: Disjoint cycle decomposition

Now trace cycles in this permutation:

* Start at `0`:
  `0 → 2 → 3 → 1 → 4 → 0`
  That’s one cycle: **(0 2 3 1 4)**

That covers all indices → only **1 cycle of length 5**.

---

## 🔹 Step 4: Minimum swaps

Formula:

$$
\text{min\_swaps} = n - \text{number of cycles}
$$

Here: $n = 5$, cycles = 1

$$
\text{min\_swaps} = 5 - 1 = 4
$$

So it takes **4 swaps**.

---

## 🔹 Step 5: Parity

* A cycle of length $k$ has parity = $(k-1) \bmod 2$.
* Here $k=5$, so parity = $4 \bmod 2 = 0$.
* That means this permutation is **even**.

Check with inversions:
Array `[3,5,4,2,1]` inversions are:

* (3,2), (3,1), (5,4), (5,2), (5,1), (4,2), (4,1), (2,1) → total **8 inversions**.
  Even ✅ matches cycle result.

---

## 🔹 Step 6: Summary

* **Permutation mapping:** `[0→2, 1→4, 2→3, 3→1, 4→0]`
* **Cycle decomposition:** (0 2 3 1 4)
* **Number of cycles:** 1
* **Min swaps to sort:** 4
* **Parity:** Even permutation

---

✅ So, `[3,5,4,2,1]` can be sorted in **4 swaps**.

---

Do you want me to also **show the actual swaps sequence** (which elements swap at each step) that achieves the minimum?
