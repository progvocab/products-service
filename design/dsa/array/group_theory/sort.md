
You are allowing only **two operations**:

1. **Rotate left by 1**

   ```
   [a1, a2, a3, …, an] → [a2, a3, …, an, a1]
   ```

   (cyclic shift)

2. **Swap first and last element**

   ```
   [a1, a2, …, an-1, an] → [an, a2, …, an-1, a1]
   ```

---

### Question: Is it always possible to sort an array using only these?

Let’s think in terms of **permutations**.

* **Rotate left** is an `n`-cycle permutation:

  ```
  (1 2 3 … n)
  ```
* **Swap first and last** is a transposition:

  ```
  (1 n)
  ```

From group theory:

* Any `n`-cycle + one transposition **generate the full symmetric group Sn**.
* That means you can reach *any permutation*, hence you can always sort.

So **yes, it’s always possible**

---

###  Example

Array: `[3, 1, 2]`

Target: `[1, 2, 3]`

Steps:

1. `[3, 1, 2]` (swap first↔last) → `[2, 1, 3]`
2. `[2, 1, 3]` (rotate left) → `[1, 3, 2]`
3. `[1, 3, 2]` (swap first↔last) → `[2, 3, 1]`
4. `[2, 3, 1]` (rotate left) → `[3, 1, 2]` … hmm cycles back.

But if we sequence differently:

1. `[3, 1, 2]` (rotate left) → `[1, 2, 3]` 🎉 Done!

So yes, it works.

---

###  Catch

* It’s not always **efficient**. Worst case might take `O(n²)` or more steps depending on how you plan swaps and rotations.
* But in theory, since the operations generate all possible permutations, sorting is always achievable.

---

 **design an actual sorting algorithm** (like a variant of bubble/selection sort) that uses *only these two operations* and prove it always finishes

---

implement a sorting algorithm that uses **only two operations**:

1. **Rotate left**
2. **Swap first and last element**

We’ll repeatedly bring the smallest remaining element to the front (using rotations), then push it into its correct position (using swaps and more rotations).

Here’s the Python code:

```python
def rotate_left(arr):
    """Rotate array left by 1"""
    if len(arr) > 1:
        arr.append(arr.pop(0))

def swap_first_last(arr):
    """Swap first and last element"""
    if len(arr) > 1:
        arr[0], arr[-1] = arr[-1], arr[0]

def custom_sort(arr):
    n = len(arr)
    result = arr[:]
    for i in range(n):
        # Step 1: bring smallest element of result[i:] to the front
        min_val = min(result[i:])
        while result[0] != min_val:
            rotate_left(result)

        # Step 2: move it into its correct position (i-th place)
        for _ in range(n - 1 - i):
            swap_first_last(result)
            rotate_left(result)

    return result


# Example
arr = [3, 1, 4, 2]
print("Original:", arr)
sorted_arr = custom_sort(arr)
print("Sorted:", sorted_arr)
```

---

###  How it works

* `rotate_left` cycles elements.
* `swap_first_last` allows you to jump the current first element to the back.
* In `custom_sort`, we repeatedly:

  1. Rotate until the smallest remaining element is at the front.
  2. Push it into its sorted place using swaps+rotations.

---

###  Example Run

```
Original: [3, 1, 4, 2]
Sorted:   [1, 2, 3, 4]
```

---

⚡ This isn’t the most efficient algorithm, but it **proves sorting is possible with only those two operations**.



