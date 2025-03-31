### **Algorithms with O(log N) Complexity (Logarithmic Time Complexity)**  

Logarithmic time complexity **O(log N)** means that as the input size **N** grows, the execution time grows proportionally to the logarithm of **N**. This often happens when an algorithm **divides the problem into smaller parts** at each step.

---

## **1️⃣ Binary Search (O(log N))**
- **Concept**: Repeatedly divides the sorted array into half until the target element is found.
- **Best Case**: O(1) (when the element is at the middle).
- **Worst Case**: O(log N).
- **Code Example (Recursive & Iterative):**
  ```python
  def binary_search(arr, target, low, high):
      if low > high:
          return -1  # Not found
      mid = (low + high) // 2
      if arr[mid] == target:
          return mid
      elif arr[mid] < target:
          return binary_search(arr, target, mid + 1, high)
      else:
          return binary_search(arr, target, low, mid - 1)

  arr = [1, 3, 5, 7, 9, 11, 13]
  print(binary_search(arr, 7, 0, len(arr) - 1))  # Output: 3
  ```

  **Iterative Version:**
  ```python
  def binary_search_iterative(arr, target):
      low, high = 0, len(arr) - 1
      while low <= high:
          mid = (low + high) // 2
          if arr[mid] == target:
              return mid
          elif arr[mid] < target:
              low = mid + 1
          else:
              high = mid - 1
      return -1  # Not found
  ```

---

## **2️⃣ Binary Heap Operations (O(log N))**
- **Concept**: A **binary heap** is a complete binary tree where each parent node has a priority-based relationship with children.
- **Insertion & Deletion (Heapify Up/Down) → O(log N)**
- **Code Example:**
  ```python
  import heapq

  heap = []
  heapq.heappush(heap, 5)
  heapq.heappush(heap, 3)
  heapq.heappush(heap, 8)

  print(heapq.heappop(heap))  # Output: 3 (Min element is removed)
  ```

---

## **3️⃣ Balanced Search Trees (O(log N))**
### **3.1 AVL Tree / Red-Black Tree**
- **Concept**: Keeps a balanced structure so that search, insert, and delete remain O(log N).
- **Example:**
  ```python
  class Node:
      def __init__(self, key):
          self.key = key
          self.left = None
          self.right = None

  def insert(root, key):
      if root is None:
          return Node(key)
      if key < root.key:
          root.left = insert(root.left, key)
      else:
          root.right = insert(root.right, key)
      return root  # (Rebalancing would be needed for AVL/Red-Black Trees)
  ```

---

## **4️⃣ Divide and Conquer Algorithms (O(log N))**
### **4.1 Fast Exponentiation (O(log N))**
- **Concept**: Instead of multiplying **x** by itself **n** times, we use **Divide and Conquer**.
- **Example:**
  ```python
  def fast_exponentiation(x, n):
      if n == 0:
          return 1
      elif n % 2 == 0:
          half = fast_exponentiation(x, n // 2)
          return half * half
      else:
          return x * fast_exponentiation(x, n - 1)

  print(fast_exponentiation(2, 10))  # Output: 1024
  ```

---

## **5️⃣ Binary Indexed Tree / Fenwick Tree (O(log N))**
- **Concept**: Used for **efficient range queries** in arrays.
- **Operations**:
  - **Update → O(log N)**
  - **Query (Prefix Sum) → O(log N)**
- **Example:**
  ```python
  class FenwickTree:
      def __init__(self, size):
          self.size = size
          self.tree = [0] * (size + 1)

      def update(self, index, value):
          while index <= self.size:
              self.tree[index] += value
              index += index & -index  # Move to next index

      def query(self, index):
          sum = 0
          while index > 0:
              sum += self.tree[index]
              index -= index & -index  # Move to parent index
          return sum

  ft = FenwickTree(5)
  ft.update(1, 3)
  ft.update(2, 2)
  print(ft.query(2))  # Output: 5 (Sum of first 2 elements)
  ```

---

## **6️⃣ Logarithmic Search Algorithms (O(log N))**
### **6.1 Interpolation Search**
- **Concept**: Like **binary search**, but assumes data is uniformly distributed.
- **Code Example:**
  ```python
  def interpolation_search(arr, target):
      low, high = 0, len(arr) - 1
      while low <= high and target >= arr[low] and target <= arr[high]:
          pos = low + ((target - arr[low]) * (high - low) // (arr[high] - arr[low]))
          if arr[pos] == target:
              return pos
          elif arr[pos] < target:
              low = pos + 1
          else:
              high = pos - 1
      return -1

  arr = [10, 20, 30, 40, 50]
  print(interpolation_search(arr, 30))  # Output: 2
  ```

---

## **7️⃣ Logarithmic Sorting Algorithms (O(N log N))**
### **7.1 Merge Sort**
- **Concept**: Recursively divides the array into two halves, sorts them, and merges them.
- **Code Example:**
  ```python
  def merge_sort(arr):
      if len(arr) <= 1:
          return arr
      mid = len(arr) // 2
      left = merge_sort(arr[:mid])
      right = merge_sort(arr[mid:])
      return merge(left, right)

  def merge(left, right):
      result = []
      i = j = 0
      while i < len(left) and j < len(right):
          if left[i] < right[j]:
              result.append(left[i])
              i += 1
          else:
              result.append(right[j])
              j += 1
      result.extend(left[i:])
      result.extend(right[j:])
      return result

  arr = [5, 3, 8, 1, 2]
  print(merge_sort(arr))  # Output: [1, 2, 3, 5, 8]
  ```

---

## **💡 Summary of O(log N) Algorithms**
| **Algorithm**                 | **Time Complexity** |
|--------------------------------|---------------------|
| **Binary Search**              | O(log N) |
| **Heap Insert/Delete (Heapify)** | O(log N) |
| **Balanced Trees (AVL, Red-Black)** | O(log N) |
| **Fast Exponentiation**        | O(log N) |
| **Fenwick Tree Operations**    | O(log N) |
| **Interpolation Search**       | O(log N) |
| **Merge Sort**                 | O(N log N) |
| **Quick Sort (Average Case)**  | O(N log N) |

---

### **🚀 Optimization Strategies**
1. **Convert O(N) operations to O(log N) using Binary Search or Trees.**
2. **Use divide and conquer techniques (e.g., Fast Exponentiation, Merge Sort).**
3. **Utilize special data structures (e.g., Fenwick Tree, Heap, Balanced Trees).**

Would you like examples of how to optimize **O(N²) to O(N log N)**? 🚀