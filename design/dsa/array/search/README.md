# **Array search algorithms**   


##  1. **Linear Search**

```python
def linear_search(arr, target):
    for i, x in enumerate(arr):
        if x == target:
            return i
    return -1
```

 **O(n)** — Works on unsorted arrays.

 

##  2. **Binary Search (Iterative)**

```python
def binary_search(arr, target):
    l, r = 0, len(arr) - 1
    while l <= r:
        mid = (l + r) // 2
        if arr[mid] == target: return mid
        elif arr[mid] < target: l = mid + 1
        else: r = mid - 1
    return -1
```

  **O(log n)** — Works only on **sorted** arrays.

 

##  3. **Binary Search (Recursive)**

```python
def binary_search_rec(arr, target, l=0, r=None):
    if r is None: r = len(arr) - 1
    if l > r: return -1
    mid = (l + r) // 2
    if arr[mid] == target: return mid
    if arr[mid] < target: return binary_search_rec(arr, target, mid + 1, r)
    return binary_search_rec(arr, target, l, mid - 1)
```

For Dynamic arrays `Binary Search Tree` gives better performance than `Binary Search` , since adding a new element leads to sorting the entire array O(n)
<br/>

[Tree](/design/dsa/tree/)

###   **Comparision**

| Operation  | **Sorted Array** | **AVL Tree** | **Red-Black Tree** |
| ---------- | ---------------- | ------------ | ------------------ |
| **Search** | O(log n)         | O(log n)     | O(log n)           |
| **Insert** | O(n)             | O(log n)     | O(log n)           |
| **Delete** | O(n)             | O(log n)     | O(log n)           |

---

###   **Details**

| **Operation**          | **Sorted Array**                  | **AVL Tree**                        | **Red-Black Tree**                         | **Explanation**                                                                                     |
| ---------------------- | --------------------------------- | ----------------------------------- | ------------------------------------------ | --------------------------------------------------------------------------------------------------- |
| **Search**             | O(log n)                          | O(log n)                            | O(log n)                                   | All support binary search logic; trees maintain balance ensuring logarithmic height.                |
| **Insert**             | O(n)                              | O(log n)                            | O(log n)                                   | Sorted array needs shifting to maintain order. Trees rebalance but cost stays logarithmic.          |
| **Delete**             | O(n)                              | O(log n)                            | O(log n)                                   | Deletion in array requires shifting. In trees, balance is restored via rotations.                   |
| **Balance Strictness** | Always sorted, no structure       | Very strict (height difference ≤ 1) | Less strict (color rules maintain balance) | AVL ensures tighter balance, so faster search but more rebalancing cost.                            |
| **Rotations Needed**   |   None                            |   Sometimes (1–2 rotations)         |   Sometimes (1 rotation avg)               | Trees rebalance via rotations after insert/delete.                                                  |
| **Memory Usage**       | Compact (contiguous array)        | Higher (pointers + balance factor)  | Higher (pointers + color bit)              | Trees use extra memory per node.                                                                    |
| **Cache Locality**     | Excellent                         | Poor                                | Poor                                       | Arrays are contiguous → CPU cache-friendly. Trees are pointer-based → scattered memory.             |
| **Best Use Case**      | Mostly reads, few inserts/deletes | Lookup-heavy + frequent updates     | Balanced performance for mixed workloads   | Sorted arrays are great for read-heavy, static data. Trees are better when structure changes often. |



 

### Other Search Algorithms

## 4. **Jump Search**

```python
import math
def jump_search(arr, target):
    n, step = len(arr), int(math.sqrt(len(arr)))
    prev = 0
    while prev < n and arr[min(step, n) - 1] < target:
        prev, step = step, step + int(math.sqrt(n))
    for i in range(prev, min(step, n)):
        if arr[i] == target:
            return i
    return -1
```

 **O(√n)** — Best for sorted arrays.

---

##  5. **Interpolation Search**

```python
def interpolation_search(arr, target):
    low, high = 0, len(arr) - 1
    while low <= high and arr[low] <= target <= arr[high]:
        pos = low + (high - low) * (target - arr[low]) // (arr[high] - arr[low])
        if arr[pos] == target: return pos
        if arr[pos] < target: low = pos + 1
        else: high = pos - 1
    return -1
```

Works best on **uniformly distributed sorted arrays**.

---

##  6. **Exponential Search**

```python
def exponential_search(arr, target):
    if arr[0] == target: return 0
    i = 1
    while i < len(arr) and arr[i] <= target:
        i *= 2
    l, r = i // 2, min(i, len(arr) - 1)
    # Binary search in the found range
    while l <= r:
        mid = (l + r) // 2
        if arr[mid] == target: return mid
        elif arr[mid] < target: l = mid + 1
        else: r = mid - 1
    return -1
```

 Efficient when the **target is near the start**.

---

##  7. **Ternary Search**

```python
def ternary_search(arr, target):
    l, r = 0, len(arr) - 1
    while l <= r:
        mid1 = l + (r - l) // 3
        mid2 = r - (r - l) // 3
        if arr[mid1] == target: return mid1
        if arr[mid2] == target: return mid2
        if target < arr[mid1]: r = mid1 - 1
        elif target > arr[mid2]: l = mid2 + 1
        else: l, r = mid1 + 1, mid2 - 1
    return -1
```

 For **sorted unimodal** arrays.

---

##  8. **Fibonacci Search**

```python
def fibonacci_search(arr, target):
    fibMMm2, fibMMm1 = 0, 1
    fibM = fibMMm2 + fibMMm1
    n = len(arr)
    while fibM < n:
        fibMMm2, fibMMm1 = fibMMm1, fibM
        fibM = fibMMm2 + fibMMm1
    offset = -1
    while fibM > 1:
        i = min(offset + fibMMm2, n - 1)
        if arr[i] < target:
            fibM, fibMMm1, fibMMm2 = fibMMm1, fibMMm2, fibM - fibMMm1
            offset = i
        elif arr[i] > target:
            fibM, fibMMm1, fibMMm2 = fibMMm2, fibMMm1 - fibMMm2, fibM - fibMMm2
        else:
            return i
    if fibMMm1 and offset + 1 < n and arr[offset + 1] == target:
        return offset + 1
    return -1
```

 Alternative to binary search — uses **Fibonacci numbers**.

---

##  Summary Table

| Algorithm      | Works on Sorted? | Avg Time     | Space | Notes                   |
| -------------- | ---------------- | ------------ | ----- | ----------------------- |
| Linear Search  | ❌                | O(n)         | O(1)  | Simple, for unsorted    |
| Binary Search  | ✅                | O(log n)     | O(1)  | Divide & conquer        |
| Jump Search    | ✅                | O(√n)        | O(1)  | Uses block jumps        |
| Interpolation  | ✅                | O(log log n) | O(1)  | Uniform distribution    |
| Exponential    | ✅                | O(log n)     | O(1)  | Fast growing step       |
| Ternary Search | ✅                | O(log₃ n)    | O(1)  | Unimodal arrays         |
| Fibonacci      | ✅                | O(log n)     | O(1)  | Binary alt, no division |

---

