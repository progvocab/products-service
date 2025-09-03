### 🔷 K-Way Merge Sort – Explained with Example

**K-way merge sort** is an extension of the standard merge sort. Instead of dividing the array into 2 parts (as in regular merge sort), we divide it into **k parts**, sort each part, and then **merge the k sorted parts**.

It’s especially useful for:

* **External sorting** (e.g., when data doesn't fit into memory)
* **Merging k sorted lists** (e.g., in search engines or databases)

---

## 🔹 Concept

* **Input**: An unsorted array
* **Step 1**: Divide the array into `k` parts
* **Step 2**: Recursively sort each part using k-way merge sort
* **Step 3**: Merge the `k` sorted parts into a single sorted array using a **min-heap (priority queue)** for efficiency

---

## 🔹 Real-World Use Case

Suppose you have **10 GB of log files**, but your memory can only hold 1 GB. You:

1. Break the file into 10 chunks (1 GB each)
2. Sort each chunk in memory (using regular sort), store them as sorted files on disk
3. Then **k-way merge** the 10 sorted files using a min-heap

---

## 🔹 Example

Let’s say you have 3 sorted arrays:

```
A = [1, 5, 9]
B = [2, 6, 8]
C = [0, 4, 7]
```

We want to **merge these 3 arrays into one sorted array**.

---

### 🔹 Min-Heap Based K-Way Merge

**Step-by-step**:

1. Insert the first element of each array into a min-heap:

   ```
   Heap = [(0, C), (1, A), (2, B)]  → Heap orders by value
   ```

2. Extract min (0), push next element from same array (C):

   ```
   Result = [0]
   Heap = [(1, A), (2, B), (4, C)]
   ```

3. Extract min (1), push next from A:

   ```
   Result = [0, 1]
   Heap = [(4, C), (2, B), (5, A)]
   ```

... and so on, until the heap is empty.

**Final Merged Array**:

```
[0, 1, 2, 4, 5, 6, 7, 8, 9]
```

---

## 🔹 Python Code for K-Way Merge

```python
import heapq

def k_way_merge(sorted_lists):
    min_heap = []
    result = []

    # Initialize heap with first element from each list
    for i, lst in enumerate(sorted_lists):
        if lst:
            heapq.heappush(min_heap, (lst[0], i, 0))  # (value, list_index, element_index)

    while min_heap:
        val, list_idx, element_idx = heapq.heappop(min_heap)
        result.append(val)

        # If there's a next element in the same list, add it to heap
        if element_idx + 1 < len(sorted_lists[list_idx]):
            next_tuple = (sorted_lists[list_idx][element_idx + 1], list_idx, element_idx + 1)
            heapq.heappush(min_heap, next_tuple)

    return result

# Example usage
lists = [
    [1, 5, 9],
    [2, 6, 8],
    [0, 4, 7]
]

print(k_way_merge(lists))  # Output: [0, 1, 2, 4, 5, 6, 7, 8, 9]
```

---

## 🔹 Time and Space Complexity

* **Time Complexity**: `O(N log k)`

  * `N`: total number of elements
  * `k`: number of lists
* **Space Complexity**: `O(k)` for the heap

---

Would you like the same in Java or with external files (e.g., merging large sorted files from disk)?
