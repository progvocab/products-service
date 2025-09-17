Here's a **one-liner** version of **binary search** in Python (returns the index or -1 if not found):

```python
binary_search = lambda arr, x: next((i for i in range(len(arr)) if arr[i] == x), -1) if x in arr else -1
```

However, **this is not true binary search** — it’s linear search with a clever trick.
Let’s make a **real one-liner binary search** using `bisect` (which is a true binary search under the hood):

---

### ✅ True Binary Search (One-liner using `bisect`)

```python
import bisect
binary_search = lambda arr, x: i if (i := bisect.bisect_left(arr, x)) < len(arr) and arr[i] == x else -1
```

#### 🔍 Example Usage:

```python
arr = [1, 3, 5, 7, 9, 11]
print(binary_search(arr, 7))   # Output: 3
print(binary_search(arr, 8))   # Output: -1
```

---

Would you like a one-liner version of recursive or iterative binary search without using built-in modules like `bisect`?
