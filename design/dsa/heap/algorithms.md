 

##   **Heap**

A **Heap** is a **complete binary tree** (all levels are filled except possibly the last, which is filled left to right) that satisfies the **heap property**:

* **Min-Heap** → The **parent** node is **smaller** than or equal to its children.
* **Max-Heap** → The **parent** node is **greater** than or equal to its children.

It is commonly implemented as an **array** (not linked nodes).
 

##   **Key Points**

| Feature                      | Description                                           |
| ---------------------------- | ----------------------------------------------------- |
| **Type**                     | Binary tree (Complete)                                |
| **Order Property**           | Parent ≤ children (min-heap) or ≥ (max-heap)          |
| **Shape Property**           | Complete — filled from left to right                  |
| **Storage**                  | Implemented as an array                               |
| **Root**                     | Contains the min (min-heap) or max (max-heap) element |
| **Children Index (0-based)** | Left: `2*i + 1`, Right: `2*i + 2`                     |
| **Parent Index**             | `(i - 1) // 2`                                        |

 

##   **Time Complexities**

| Operation                 | Complexity | Description                             |
| ------------------------- | ---------- | --------------------------------------- |
| **Insert (push)**         | O(log n)   | May bubble up to maintain heap property |
| **Extract Min/Max (pop)** | O(log n)   | Remove root and restructure             |
| **Peek (get min/max)**    | O(1)       | Access root element                     |
| **Heapify (build heap)**  | O(n)       | Convert list into heap                  |

 

##   **Python Implementation using `heapq`**
 

```python
import heapq

# Create an empty heap
heap = []

# Insert elements (push)
heapq.heappush(heap, 10)
heapq.heappush(heap, 5)
heapq.heappush(heap, 30)
heapq.heappush(heap, 2)

print("Min-Heap:", heap)  # Internally stored as [2, 5, 30, 10]

# Get the smallest element (root)
print("Smallest element:", heap[0])

# Remove smallest element (pop)
min_elem = heapq.heappop(heap)
print("Removed:", min_elem)
print("Heap after removal:", heap)

# Convert an existing list to a heap (heapify)
nums = [9, 4, 7, 1, -2, 6, 5]
heapq.heapify(nums)
print("Heapified list:", nums)

# Get n smallest/largest elements
print("3 smallest:", heapq.nsmallest(3, nums))
print("2 largest:", heapq.nlargest(2, nums))
```

---

###   **Output Example**

```
Min-Heap: [2, 5, 30, 10]
Smallest element: 2
Removed: 2
Heap after removal: [5, 10, 30]
Heapified list: [-2, 1, 5, 4, 9, 6, 7]
3 smallest: [-2, 1, 4]
2 largest: [9, 7]
```

 

##   **Applications of Heap**

| Use Case                 | Description                                     |
| ------------------------ | ----------------------------------------------- |
| **Priority Queues**      | Tasks/jobs with priorities (OS scheduling).     |
| **Heap Sort**            | Sorting with O(n log n).                        |
| **Dijkstra’s Algorithm** | Finding shortest paths (min-heap of distances). |
| **Median of Stream**     | Maintain two heaps (max and min).               |
| **Top K elements**       | Efficiently get largest or smallest K elements. |

---

##   **Visualization Example (Min-Heap)**

```
        2
      /   \
     5     30
    /
   10
```
 

#   1. **Kth Largest Element in an Array**

**Problem:** Given an unsorted array, find the `k`-th largest element.

  Use a **Min-Heap** of size `k`.

```python
import heapq

def kth_largest(nums, k):
    heap = nums[:k]
    heapq.heapify(heap)
    for num in nums[k:]:
        if num > heap[0]:
            heapq.heapreplace(heap, num)
    return heap[0]

print(kth_largest([3,2,1,5,6,4], 2))  # Output: 5
```

* **Why heap?** Keeps only `k` largest elements → O(n log k).

---

#   2. **Top K Frequent Elements**

**Problem:** Given an array of numbers, return the `k` most frequent elements.

  Use a **Heap** sorted by frequency.

```python
import heapq
from collections import Counter

def top_k_frequent(nums, k):
    freq = Counter(nums)
    return [item for item, _ in heapq.nlargest(k, freq.items(), key=lambda x: x[1])]

print(top_k_frequent([1,1,1,2,2,3], 2))  # Output: [1, 2]
```

* **Why heap?** Efficient way to get `k` max frequencies without sorting everything.

---

#   3. **Merge K Sorted Lists**

**Problem:** Merge `k` sorted linked lists into one sorted list.

  Use a **Min-Heap** to always pick the smallest head node.

```python
import heapq

def merge_k_sorted_lists(lists):
    heap = []
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (lst[0], i, 0))

    result = []
    while heap:
        val, li, idx = heapq.heappop(heap)
        result.append(val)
        if idx + 1 < len(lists[li]):
            heapq.heappush(heap, (lists[li][idx + 1], li, idx + 1))
    return result

print(merge_k_sorted_lists([[1,4,5],[1,3,4],[2,6]]))
# Output: [1,1,2,3,4,4,5,6]
```

* **Why heap?** Avoids merging lists all at once → O(N log k).

---

#   4. **Connect Ropes with Minimum Cost**

**Problem:** Given lengths of ropes, connect them into one rope with minimum cost (cost = sum of lengths connected).

  Use a **Min-Heap** to always pick two smallest ropes.

```python
import heapq

def min_cost_to_connect(ropes):
    heapq.heapify(ropes)
    total = 0
    while len(ropes) > 1:
        first = heapq.heappop(ropes)
        second = heapq.heappop(ropes)
        cost = first + second
        total += cost
        heapq.heappush(ropes, cost)
    return total

print(min_cost_to_connect([1,2,3,4,5]))  # Output: 33
```

* **Why heap?** Greedy choice with smallest ropes ensures minimal total cost.

---

#   5. **Sliding Window Maximum**

**Problem:** Find maximum in every sliding window of size `k`.

  Use a **Max-Heap** (or monotonic deque for optimal).

```python
import heapq

def sliding_window_max(nums, k):
    heap, result = [], []
    for i, n in enumerate(nums):
        heapq.heappush(heap, (-n, i))  # store negative for max-heap
        while heap[0][1] <= i - k:
            heapq.heappop(heap)
        if i >= k - 1:
            result.append(-heap[0][0])
    return result

print(sliding_window_max([1,3,-1,-3,5,3,6,7], 3))
# Output: [3,3,5,5,6,7]
```

* **Why heap?** Efficiently tracks max element in window.

---

# 📊 Summary

| Problem                 | Heap Type           | Key Idea                   |
| ----------------------- | ------------------- | -------------------------- |
| Kth Largest Element     | Min-Heap (size k)   | Keep track of top k        |
| Top K Frequent Elements | Max-Heap / Min-Heap | Order by frequency         |
| Merge K Sorted Lists    | Min-Heap            | Pick smallest across lists |
| Connect Ropes Min Cost  | Min-Heap            | Greedy merge smallest      |
| Sliding Window Maximum  | Max-Heap            | Track max in window        |

---

👉 These are all **medium-level** problems where heaps give a big efficiency boost over brute-force.

Would you like me to also show you **Java versions** of these heap problems (using `PriorityQueue`), since you often work in Java + Spring Boot?
