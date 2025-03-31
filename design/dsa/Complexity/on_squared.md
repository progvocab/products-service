### **Algorithms with O(N²) Complexity (Quadratic Time Complexity)**  

Quadratic time complexity **O(N²)** means that as the input size **N** grows, the execution time increases proportionally to **N²**. This often happens when an algorithm has **nested loops**.

---

### **1️⃣ Sorting Algorithms (O(N²))**
#### **1.1 Bubble Sort**
- **Concept**: Repeatedly swaps adjacent elements if they are in the wrong order.
- **Worst Case**: O(N²) (when array is in reverse order).
- **Code Example:**
  ```python
  def bubble_sort(arr):
      n = len(arr)
      for i in range(n):
          for j in range(0, n - i - 1):
              if arr[j] > arr[j + 1]:
                  arr[j], arr[j + 1] = arr[j + 1], arr[j]
      return arr

  arr = [5, 3, 8, 1, 2]
  print(bubble_sort(arr))  # Output: [1, 2, 3, 5, 8]
  ```

---

#### **1.2 Insertion Sort**
- **Concept**: Builds the sorted list one element at a time, shifting elements as needed.
- **Worst Case**: O(N²) (when array is sorted in reverse).
- **Code Example:**
  ```python
  def insertion_sort(arr):
      for i in range(1, len(arr)):
          key = arr[i]
          j = i - 1
          while j >= 0 and key < arr[j]:
              arr[j + 1] = arr[j]
              j -= 1
          arr[j + 1] = key
      return arr

  arr = [5, 3, 8, 1, 2]
  print(insertion_sort(arr))  # Output: [1, 2, 3, 5, 8]
  ```

---

#### **1.3 Selection Sort**
- **Concept**: Selects the smallest element and places it in the correct position.
- **Worst Case**: O(N²).
- **Code Example:**
  ```python
  def selection_sort(arr):
      n = len(arr)
      for i in range(n):
          min_idx = i
          for j in range(i + 1, n):
              if arr[j] < arr[min_idx]:
                  min_idx = j
          arr[i], arr[min_idx] = arr[min_idx], arr[i]
      return arr

  arr = [5, 3, 8, 1, 2]
  print(selection_sort(arr))  # Output: [1, 2, 3, 5, 8]
  ```

---

### **2️⃣ Brute Force Algorithms (O(N²))**
#### **2.1 Two Sum (Brute Force)**
- **Concept**: Checks all pairs to see if they sum up to the target.
- **Better Approach**: Use HashMap (O(N)).
- **Code Example:**
  ```python
  def two_sum(nums, target):
      n = len(nums)
      for i in range(n):
          for j in range(i + 1, n):
              if nums[i] + nums[j] == target:
                  return [i, j]
      return []

  print(two_sum([2, 7, 11, 15], 9))  # Output: [0, 1]
  ```

---

#### **2.2 Longest Common Substring (Naive Approach)**
- **Concept**: Compares all possible substrings of two strings.
- **Better Approach**: Use Dynamic Programming (O(N*M)).
- **Code Example:**
  ```python
  def longest_common_substring(str1, str2):
      max_length = 0
      for i in range(len(str1)):
          for j in range(len(str2)):
              l = 0
              while (i + l < len(str1)) and (j + l < len(str2)) and (str1[i + l] == str2[j + l]):
                  l += 1
              max_length = max(max_length, l)
      return max_length

  print(longest_common_substring("abcde", "abfde"))  # Output: 2 (for "ab")
  ```

---

### **3️⃣ Graph Algorithms (O(N²))**
#### **3.1 Floyd-Warshall Algorithm (All-Pairs Shortest Path)**
- **Concept**: Finds shortest paths between all pairs of vertices.
- **Time Complexity**: O(V³) for dense graphs, but O(N²) if N ≈ V.
- **Code Example:**
  ```python
  INF = float('inf')

  def floyd_warshall(graph):
      V = len(graph)
      dist = [row[:] for row in graph]

      for k in range(V):
          for i in range(V):
              for j in range(V):
                  dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
      return dist

  graph = [
      [0, 3, INF, INF],
      [2, 0, INF, INF],
      [INF, 7, 0, 1],
      [6, INF, INF, 0]
  ]

  print(floyd_warshall(graph))
  ```

---

#### **3.2 Adjacency Matrix Representation of Graph**
- **Concept**: Representing a graph using an adjacency matrix takes O(N²) space.
- **Example:**
  ```python
  def create_adjacency_matrix(n, edges):
      adj_matrix = [[0] * n for _ in range(n)]
      for u, v in edges:
          adj_matrix[u][v] = 1
          adj_matrix[v][u] = 1  # For undirected graphs
      return adj_matrix

  edges = [(0, 1), (1, 2), (2, 3)]
  print(create_adjacency_matrix(4, edges))
  ```

---

### **4️⃣ Nested Loops in Brute Force Approaches**
#### **4.1 Checking for Duplicate Pairs**
- **Concept**: Iterates over all pairs to check for duplicates.
- **Code Example:**
  ```python
  def has_duplicates(arr):
      n = len(arr)
      for i in range(n):
          for j in range(i + 1, n):
              if arr[i] == arr[j]:
                  return True
      return False

  print(has_duplicates([1, 2, 3, 4, 5]))  # Output: False
  print(has_duplicates([1, 2, 3, 3, 5]))  # Output: True
  ```

---

### **💡 Summary**
| **Algorithm**                | **Time Complexity** |
|------------------------------|--------------------|
| **Bubble Sort**               | O(N²) |
| **Insertion Sort**            | O(N²) |
| **Selection Sort**            | O(N²) |
| **Brute Force Two Sum**       | O(N²) |
| **Longest Common Substring**  | O(N²) |
| **Floyd-Warshall Algorithm**  | O(N²) for specific cases |
| **Adjacency Matrix Creation** | O(N²) |

---

### **🚀 Optimization Strategies**
1. **Use Hashing:** Convert O(N²) search into O(N).
2. **Use Divide and Conquer:** E.g., QuickSort, Merge Sort.
3. **Use Dynamic Programming:** Avoid recomputation (e.g., LCS problem).
4. **Use Data Structures:** Binary Search Tree or HashMap can improve lookup performance.

Would you like examples of how to optimize these to **O(N log N)** or **O(N)**? 🚀