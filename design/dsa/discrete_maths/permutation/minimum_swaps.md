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
