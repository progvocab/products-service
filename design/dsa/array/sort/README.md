**Most common sorting algorithms**



##  1. **Bubble Sort** (Simple but inefficient)

* Repeatedly swaps adjacent elements if they are in the wrong order.
* Best for educational purposes.

###   Python Code

```python
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):  # Last i elements are already sorted
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
```

⏱ Time: O(n²)
📦 Space: O(1)

---

## ✅ 2. **Selection Sort**

* Repeatedly finds the minimum element and moves it to the front.

### 🔧 Python Code

```python
def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i+1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
```

⏱ Time: O(n²)
📦 Space: O(1)

---

## ✅ 3. **Insertion Sort**

* Builds sorted array one item at a time.

### 🔧 Python Code

```python
def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j+1] = arr[j]
            j -= 1
        arr[j+1] = key
```

⏱ Time: O(n²), Best: O(n) if already sorted
📦 Space: O(1)

---

## ✅ 4. **Merge Sort** (Divide and Conquer, stable)

* Recursively divides and merges arrays.
* Good for large data.

### 🔧 Python Code

```python
def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        L = arr[:mid]
        R = arr[mid:]

        merge_sort(L)
        merge_sort(R)

        i = j = k = 0

        # Merge the sorted halves
        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1

        # Remaining elements
        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1
        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1
```

⏱ Time: O(n log n)
📦 Space: O(n)

---

## ✅ 5. **Quick Sort** (Fastest in practice, divide & conquer)

* Picks a **pivot**, partitions the array, and recursively sorts.
* Not stable, but fast.

### 🔧 Python Code

```python
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[0]
    less = [x for x in arr[1:] if x <= pivot]
    more = [x for x in arr[1:] if x > pivot]
    return quick_sort(less) + [pivot] + quick_sort(more)
```

⏱ Time: O(n log n), Worst: O(n²)
📦 Space: O(log n) average

---

## ✅ 6. **Built-in Sort (Timsort)**

```python
arr = [3, 1, 4, 1, 5]
arr.sort()  # or sorted(arr)
```

* Uses **Timsort** (Hybrid of Merge + Insertion Sort)
* Highly optimized for real-world use.

⏱ Time: O(n log n), Best: O(n)
📦 Space: O(n)



## Summary 

| Algorithm          | Best       | Average    | Worst      | Space    | Stable | Notes                           |
| ------------------ | ---------- | ---------- | ---------- | -------- | ------ | ------------------------------- |
| Bubble Sort        | O(n)       | O(n²)      | O(n²)      | O(1)     | ✅      | Educational                     |
| Selection Sort     | O(n²)      | O(n²)      | O(n²)      | O(1)     | ❌      | Simple, but slow                |
| Insertion Sort     | O(n)       | O(n²)      | O(n²)      | O(1)     | ✅      | Good for small/partially sorted |
| Merge Sort         | O(n log n) | O(n log n) | O(n log n) | O(n)     | ✅      | Stable, predictable             |
| Quick Sort         | O(n log n) | O(n log n) | O(n²)      | O(log n) | ❌      | Fastest in practice             |
| Timsort (built-in) | O(n)       | O(n log n) | O(n log n) | O(n)     | ✅      | Used in Python's sort           |



### K Way Merge 
- k-way merge of sorted arrays means merging k sorted arrays into one fully sorted array.
**Input**
```python
arr1 = [1, 4, 7]
arr2 = [2, 5, 8]
arr3 = [3, 6, 9]
```
**Output**
```python
[1, 2, 3, 4, 5, 6, 7, 8, 9]

```
```python
import heapq

def merge_k_sorted_arrays(arrays):
    heap = []
    result = []

    # Step 1: Push first element of each array into heap
    for i, arr in enumerate(arrays):
        if arr:  # if not empty
            heapq.heappush(heap, (arr[0], i, 0))  # (value, array_index, element_index)

    # Step 2: Extract min and push next element
    while heap:
        val, arr_idx, ele_idx = heapq.heappop(heap)
        result.append(val)

        # If the array has mor
```
