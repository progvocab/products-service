The `heapq` module in Python provides an implementation of the **heap queue algorithm**, also known as a **priority queue**. It uses a **min-heap** under the hood, where the smallest element is always at the front.

---

## ✅ Key Characteristics

* **Min-heap by default** (lowest value has highest priority)
* Backed by a regular Python list
* Operates in **O(log n)** time for `push` and `pop`

---

## 📦 Common Functions in `heapq`

| Function                        | Description                                                   |
| ------------------------------- | ------------------------------------------------------------- |
| `heapq.heappush(heap, item)`    | Push item onto heap, maintaining the heap invariant           |
| `heapq.heappop(heap)`           | Pop and return the smallest item from the heap                |
| `heapq.heappushpop(heap, item)` | Push new item then pop and return the smallest item           |
| `heapq.heapreplace(heap, item)` | Pop then push, more efficient when replacing root             |
| `heapq.heapify(list)`           | Convert a list into a heap in-place                           |
| `heapq.nlargest(n, iterable)`   | Return `n` largest elements from the iterable (not a heap op) |
| `heapq.nsmallest(n, iterable)`  | Return `n` smallest elements from the iterable                |

---

## 🔧 Example Usage

### ➤ Min-Heap

```python
import heapq

heap = []
heapq.heappush(heap, 4)
heapq.heappush(heap, 2)
heapq.heappush(heap, 9)
heapq.heappush(heap, 1)

print(heapq.heappop(heap))  # 1
print(heapq.heappop(heap))  # 2
```

---

### ➤ Max-Heap (Simulated with Negatives)

```python
heap = []
heapq.heappush(heap, -5)
heapq.heappush(heap, -1)
heapq.heappush(heap, -10)

print(-heapq.heappop(heap))  # 10
```

---

### ➤ Heapify

```python
nums = [5, 3, 8, 1]
heapq.heapify(nums)
print(heapq.heappop(nums))  # 1
```

---

### ➤ Find k Largest / Smallest

```python
nums = [1, 4, 2, 9, 6, 8]

print(heapq.nlargest(3, nums))   # [9, 8, 6]
print(heapq.nsmallest(2, nums))  # [1, 2]
```

---

## 🧠 Use Cases

* **Dijkstra's algorithm**
* **A* pathfinding*\*
* **Task schedulers**
* **Real-time data streams** (e.g. keep top-k smallest/largest)

---

Would you like a **custom priority queue class** using `heapq` or to simulate a **max-priority queue with objects**?
