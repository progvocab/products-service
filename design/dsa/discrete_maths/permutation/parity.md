 **parity of a permutation** is another place where group theory shows up in competitive programming.

---

# 🔹 What is Permutation Parity?

* A **permutation** can be written as a product of **swaps (transpositions)**.
* If it takes an **even number of swaps** → permutation is **even**.
* If it takes an **odd number of swaps** → permutation is **odd**.

👉 Parity is very important:

* It’s invariant: you cannot change an even permutation into odd just by rearranging swaps.
* Used in puzzles like the **15-puzzle**, Rubik’s cube, etc.

---

# 🔹 How to Compute Parity?

There are a few equivalent methods:

### 1. **Count Inversions**

* Inversion = pair `(i, j)` with `i < j` but `A[i] > A[j]`.
* If the inversion count is **even** → permutation is even.
* If odd → permutation is odd.

### 2. **Cycle Decomposition**

* Break permutation into disjoint cycles.
* Parity = `(n − number_of_cycles) mod 2`.

Both are correct. Let’s do Python code for both.

---

# ✅ Python Implementation

```python
def permutation_parity_inversions(perm):
    """Return +1 for even, -1 for odd using inversion count"""
    n = len(perm)
    inversions = 0
    for i in range(n):
        for j in range(i+1, n):
            if perm[i] > perm[j]:
                inversions += 1
    return 1 if inversions % 2 == 0 else -1


def permutation_parity_cycles(perm):
    """Return +1 for even, -1 for odd using cycle decomposition"""
    n = len(perm)
    visited = [False] * n
    cycles = 0

    for i in range(n):
        if not visited[i]:
            j = i
            while not visited[j]:
                visited[j] = True
                j = perm[j]  # follow cycle
            cycles += 1
    
    return 1 if (n - cycles) % 2 == 0 else -1


# Example permutations
perm1 = [2, 0, 1]   # permutation of [0,1,2]
perm2 = [1, 0, 2]   # just one swap of (0,1)

print("By inversions:")
print(permutation_parity_inversions(perm1))  # +1 → even
print(permutation_parity_inversions(perm2))  # -1 → odd

print("\nBy cycles:")
print(permutation_parity_cycles(perm1))      # +1 → even
print(permutation_parity_cycles(perm2))      # -1 → odd
```

---

# 🔎 Example Walkthrough

* `perm2 = [1,0,2]` (swap of first two elements):

  * Inversions = 1 → odd → parity = -1.
  * Cycles = (0 1)(2) → 3 elements, 2 cycles → `3-2 = 1` odd → parity = -1.
---
# Inversions of `[1, 0, 2]` — step-by-step

We look at all index pairs $i<j$ and check if `array[i] > array[j]`.

1. Compare index `0` (value `1`) with index `1` (value `0`):
   `1 > 0` → **inversion**. Record: index pair `(0,1)`, value pair `(1,0)`.

2. Compare index `0` (value `1`) with index `2` (value `2`):
   `1 > 2`? No → no inversion.

3. Compare index `1` (value `0`) with index `2` (value `2`):
   `0 > 2`? No → no inversion.

**Result:** there is exactly **1 inversion**:

* Index pair: `(0,1)`
* Value pair: `(1,0)`

Notes:

* The inversion count (1) equals the minimum number of adjacent swaps needed to sort the array (swap indices 0 and 1 → `[0,1,2]`).
* Since the count is odd, the permutation is an odd permutation — consistent with the cycle decomposition $(0\ 1)$ (a single transposition).
---
# Cycle decomposition of `[1, 0, 2]`

Assume the array represents the permutation $p$ on indices $0,1,2$ with $p(i)=\text{array}[i]$. So:

* $0 \mapsto 1$
* $1 \mapsto 0$
* $2 \mapsto 2$

Step-by-step:

1. Start at index `0` (not visited).

   * Visit `0`, follow $0 \mapsto 1$. Cycle so far: `[0, 1]`.
   * From `1`, follow $1 \mapsto 0$ and we return to the start → cycle closed. Cycle = `(0 1)`.

2. Move to index `1` — already visited, skip.

3. Move to index `2` (not visited).

   * Visit `2`, follow $2 \mapsto 2$. Immediately returns to itself → cycle closed. Cycle = `(2)` (a fixed point).

Final cycle decomposition: $(0\ 1)(2)$.
(Usually the fixed point `(2)` is omitted, so you may simply write $(0\ 1)$.)

Extra facts:

* Cycle lengths: 2 and 1.
* Minimum swaps to sort (if sorting to identity) = sum over cycles of $(\text{length}-1) = (2-1)+(1-1)=1$.
  Concretely, swapping positions `0` and `1` transforms `[1,0,2]` → `[0,1,2]`.


Both methods agree ✅

---


 Let’s carefully do **cycle decomposition** for the array `[4, 3, 2, 1]`.

We interpret the array as a permutation $p$ on indices $0,1,2,3$ such that:

$$
p(i) = \text{array}[i]
$$

So:

* $0 \mapsto 4$
* $1 \mapsto 3$
* $2 \mapsto 2$
* $3 \mapsto 1$

---

### Step 1: Start with index 0

* Start at `0`, follow $0 \mapsto 4$.
* From `4` … uh-oh! Our array only has indices `0,1,2,3`. So here’s the key point:

👉 If the array is meant to be a **permutation of 0..n-1**, `[4,3,2,1]` is **not valid** (because `4` is outside the range `0..3`).

But if it’s meant to be a permutation of **1..n**, then we should reinterpret:

* For **1-based indexing**:
  `[4,3,2,1]` means

  * $1 \mapsto 4$
  * $2 \mapsto 3$
  * $3 \mapsto 2$
  * $4 \mapsto 1$

---

### Step 2: Decompose into cycles (assuming 1-based indexing)

* Start at `1`:
  $1 \mapsto 4,\ 4 \mapsto 1$ → cycle: `(1 4)`

* Move to `2`:
  $2 \mapsto 3,\ 3 \mapsto 2$ → cycle: `(2 3)`

* `3` and `4` are already visited, so we’re done.

---

### Final Cycle Decomposition

$$
(1\ 4)(2\ 3)
$$

---

✅ If interpreted as a permutation of `1..n`, `[4,3,2,1]` decomposes into **two disjoint transpositions**.
❌ If interpreted as a permutation of `0..n-1`, `[4,3,2,1]` is **invalid** because of the `4`.

---

Do you want me to **always assume arrays are 1-based permutations** (like `[4,3,2,1]` → a permutation of 1..4), or should I strictly treat them as 0-based unless values go out of range?


👉 Do you want me to also show how **permutation parity is used in problems like checking solvability of the 15-puzzle** (a classic CP + math application)?

---

The parity of a permutation can be determined by counting the number of inversions or by analyzing its cycle decomposition. 
Method 1: Counting Inversions 
An inversion in a permutation $\pi$ is a pair of indices $(i, j)$ such that $i < j$ but  $\pi(i) > \pi(j)$ . A permutation is even if it has an even number of inversions, and odd if it has an odd number of inversions. 
```python 
def get_parity_inversions(permutation):
    """
    Calculates the parity of a permutation based on the number of inversions.

    Args:
        permutation: A list representing the permutation (e.g., [0, 2, 1, 3] for (0 2 1 3)).

    Returns:
        0 for even parity, 1 for odd parity.
    """
    inversions = 0
    n = len(permutation)
    for i in range(n):
        for j in range(i + 1, n):
            if permutation[i] > permutation[j]:
                inversions += 1
    return inversions % 2

# Example usage:
perm1 = [0, 2, 1, 3]  # (0 2 1 3) - inversion (2,1)
print(f"Parity of {perm1} (inversions): {get_parity_inversions(perm1)}")

perm2 = [1, 0, 3, 2]  # (1 0 3 2) - inversions (1,0), (3,2)
print(f"Parity of {perm2} (inversions): {get_parity_inversions(perm2)}")
```
Method 2: Cycle Decomposition 
Every permutation can be decomposed into disjoint cycles. The parity of a permutation is the sum of the parities of its cycles. The parity of a cycle of length $k$ is $(k-1) \pmod 2$. [1]  
```python 
def get_parity_cycles(permutation):
    """
    Calculates the parity of a permutation based on its cycle decomposition.

    Args:
        permutation: A list representing the permutation (e.g., [0, 2, 1, 3] for (0 2 1 3)).

    Returns:
        0 for even parity, 1 for odd parity.
    """
    n = len(permutation)
    visited = [False] * n
    num_cycles = 0
    total_length = 0

    for i in range(n):
        if not visited[i]:
            num_cycles += 1
            current = i
            cycle_length = 0
            while not visited[current]:
                visited[current] = True
                current = permutation[current]
                cycle_length += 1
            total_length += cycle_length

    # Parity is (n - num_cycles) % 2
    # Or, equivalently, sum of (k-1) for each cycle
    return (n - num_cycles) % 2

# Example usage:
perm1 = [0, 2, 1, 3]  # Cycles: (0)(1 2)(3) - n=4, num_cycles=3
print(f"Parity of {perm1} (cycles): {get_parity_cycles(perm1)}")

perm2 = [1, 0, 3, 2]  # Cycles: (0 1)(2 3) - n=4, num_cycles=2
print(f"Parity of {perm2} (cycles): {get_parity_cycles(perm2)}")
```
AI responses may include mistakes.

[1] https://stackoverflow.com/questions/20702782/efficiently-determine-the-parity-of-a-permutation
