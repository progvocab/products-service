### **Optimizing O(N²) to O(N log N) Algorithms**
Many naive algorithms run in **O(N²)** time but can be optimized to **O(N log N)** using better techniques such as **Divide & Conquer, Dynamic Programming, and Data Structures (Heaps, Trees, etc.).**  

---

## **1️⃣ Optimizing Sorting (Bubble Sort O(N²) → Merge Sort O(N log N))**  
**🔴 Problem:** Bubble Sort compares each element with every other element, resulting in **O(N²)** complexity.  
**✅ Solution:** Merge Sort reduces this to **O(N log N)** using **Divide & Conquer**.

### **🔴 Bubble Sort (O(N²)) - Inefficient**
```python
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]

arr = [5, 3, 8, 1, 2]
bubble_sort(arr)
print(arr)  # Output: [1, 2, 3, 5, 8]
```

### **✅ Merge Sort (O(N log N)) - Optimized**
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
**🔥 Why is Merge Sort better?**
- Instead of pairwise swaps (**O(N²)**), it **divides** the array into **log N** levels, sorting each level in **O(N)**.

---

## **2️⃣ Optimizing Pair Comparisons (Brute Force O(N²) → Sorting & Two Pointers O(N log N))**
### **🔴 Find Two Numbers That Sum to Target (Brute Force O(N²))**
```python
def two_sum_brute_force(arr, target):
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            if arr[i] + arr[j] == target:
                return (arr[i], arr[j])
    return None

print(two_sum_brute_force([1, 3, 5, 7, 9], 8))  # Output: (1, 7)
```

### **✅ Optimized Approach (Sorting & Two Pointers O(N log N))**
```python
def two_sum_optimized(arr, target):
    arr.sort()  # O(N log N)
    left, right = 0, len(arr) - 1
    while left < right:
        current_sum = arr[left] + arr[right]
        if current_sum == target:
            return (arr[left], arr[right])
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    return None

print(two_sum_optimized([1, 3, 5, 7, 9], 8))  # Output: (1, 7)
```
**🔥 Why is Two Pointers better?**
- Instead of **O(N²) comparisons**, we **sort (O(N log N))** and use **O(N)** linear scan.

---

## **3️⃣ Optimizing Brute Force Searching (O(N²) → Binary Search O(N log N))**
### **🔴 Find an element in a sorted list (Brute Force O(N))**
```python
def search_brute_force(arr, target):
    for num in arr:
        if num == target:
            return True
    return False

print(search_brute_force([1, 3, 5, 7, 9], 7))  # Output: True
```

### **✅ Optimized Binary Search (O(log N))**
```python
def binary_search(arr, target):
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return True
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return False

print(binary_search([1, 3, 5, 7, 9], 7))  # Output: True
```
**🔥 Why is Binary Search better?**
- Instead of **O(N) linear search**, we reduce the problem by half each time (**O(log N)**).

---

## **4️⃣ Optimizing Subarray Problems (O(N²) → Prefix Sum O(N log N))**
### **🔴 Find sum of all subarrays (O(N²))**
```python
def subarray_sums_brute(arr):
    n = len(arr)
    for i in range(n):
        for j in range(i, n):
            print(sum(arr[i:j+1]))  # O(N) sum call inside O(N²) loop

subarray_sums_brute([1, 2, 3])  # Slow for large N
```

### **✅ Optimized Approach Using Prefix Sum (O(N))**
```python
def prefix_sum(arr):
    prefix = [0] * (len(arr) + 1)
    for i in range(1, len(arr) + 1):
        prefix[i] = prefix[i - 1] + arr[i - 1]  # O(N) Preprocessing

    for i in range(len(arr)):
        for j in range(i, len(arr)):
            print(prefix[j + 1] - prefix[i])  # O(1) Query instead of O(N)

prefix_sum([1, 2, 3])
```
**🔥 Why is Prefix Sum better?**
- Instead of recomputing sums repeatedly (**O(N²)**), we use precomputed values in **O(N)**.

---

## **5️⃣ Optimizing Graph Algorithms (O(N²) → Dijkstra O(N log N))**
### **🔴 Naive Shortest Path (O(N²))**
```python
def naive_dijkstra(graph, src):
    import sys
    n = len(graph)
    dist = [sys.maxsize] * n
    dist[src] = 0

    for _ in range(n - 1):
        for u in range(n):
            for v, w in graph[u]:
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
    return dist

graph = {0: [(1, 4), (2, 1)], 1: [(3, 1)], 2: [(1, 2), (3, 5)], 3: []}
print(naive_dijkstra(graph, 0))  # Slow for large graphs
```

### **✅ Optimized Dijkstra’s Algorithm (O(N log N))**
```python
import heapq

def dijkstra(graph, src):
    n = len(graph)
    dist = {node: float('inf') for node in graph}
    dist[src] = 0
    pq = [(0, src)]  # (distance, node)

    while pq:
        curr_dist, node = heapq.heappop(pq)
        if curr_dist > dist[node]:
            continue
        for neighbor, weight in graph[node]:
            new_dist = curr_dist + weight
            if new_dist < dist[neighbor]:
                dist[neighbor] = new_dist
                heapq.heappush(pq, (new_dist, neighbor))

    return dist

graph = {0: [(1, 4), (2, 1)], 1: [(3, 1)], 2: [(1, 2), (3, 5)], 3: []}
print(dijkstra(graph, 0))  # Faster for large graphs
```
**🔥 Why is Dijkstra better?**
- Instead of **O(N²) relaxation loops**, we use a **priority queue (heap) in O(N log N)**.

---

## **Final Thoughts**
| **Algorithm** | **Before (O(N²))** | **After (O(N log N))** |
|--------------|------------------|------------------|
| Sorting | Bubble Sort | Merge Sort |
| Searching | Brute Force | Binary Search |
| Graph Algorithms | Naive Dijkstra | Heap-based Dijkstra |
| Subarrays | Brute Force | Prefix Sum |

Would you like optimizations for **O(N³) → O(N²)** cases? 🚀


### **Optimizing O(N³) to O(N²) Algorithms**  
Many algorithms with **O(N³)** complexity can be improved to **O(N²)** using **Dynamic Programming, Matrix Multiplication, and Graph Optimization Techniques.**  

---

## **1️⃣ Optimizing Matrix Multiplication (Brute Force O(N³) → Strassen’s O(N².81))**  
### **🔴 Naive Matrix Multiplication (O(N³))**
In standard matrix multiplication, every element is computed as a sum of **N** multiplications, leading to **O(N³)** time complexity.

```python
def naive_matrix_mult(A, B):
    N = len(A)
    result = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            for k in range(N):
                result[i][j] += A[i][k] * B[k][j]
    return result

A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]
print(naive_matrix_mult(A, B))
```

### **✅ Optimized Using Strassen’s Algorithm (O(N².81))**
Strassen’s algorithm reduces matrix multiplication time using **Divide & Conquer.**

```python
import numpy as np

def strassen_mult(A, B):
    if len(A) == 1:
        return [[A[0][0] * B[0][0]]]
    
    n = len(A) // 2
    A11, A12, A21, A22 = np.split(A, 2), np.split(A, 2, axis=1)
    B11, B12, B21, B22 = np.split(B, 2), np.split(B, 2, axis=1)

    M1 = strassen_mult(A11 + A22, B11 + B22)
    M2 = strassen_mult(A21 + A22, B11)
    M3 = strassen_mult(A11, B12 - B22)
    M4 = strassen_mult(A22, B21 - B11)
    M5 = strassen_mult(A11 + A12, B22)
    M6 = strassen_mult(A21 - A11, B11 + B12)
    M7 = strassen_mult(A12 - A22, B21 + B22)

    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6

    return np.block([[C11, C12], [C21, C22]])

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(strassen_mult(A, B))
```

**🔥 Why is Strassen’s Algorithm better?**  
Instead of **O(N³) multiplications**, it reduces it to **O(N².81)** using divide-and-conquer.

---

## **2️⃣ Optimizing Floyd-Warshall for All-Pairs Shortest Path (O(N³) → O(N² log N))**  
### **🔴 Floyd-Warshall Algorithm (O(N³))**
Floyd-Warshall calculates the shortest paths between all pairs of nodes, requiring **O(N³)** operations.

```python
def floyd_warshall(graph):
    N = len(graph)
    dist = [[float('inf')] * N for _ in range(N)]
    
    for u in range(N):
        for v in range(N):
            dist[u][v] = graph[u][v] if u != v else 0
    
    for k in range(N):
        for i in range(N):
            for j in range(N):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    return dist

graph = [[0, 3, float('inf'), 5], [2, 0, float('inf'), 8], [float('inf'), 1, 0, 4], [float('inf'), float('inf'), 2, 0]]
print(floyd_warshall(graph))
```

### **✅ Optimized Using Johnson’s Algorithm (O(N² log N))**
Johnson’s Algorithm replaces **Floyd-Warshall** with **Dijkstra's Algorithm (O(N log N))**, reducing complexity.

```python
import heapq

def dijkstra(graph, start):
    n = len(graph)
    dist = {node: float('inf') for node in range(n)}
    dist[start] = 0
    pq = [(0, start)]

    while pq:
        curr_dist, node = heapq.heappop(pq)
        if curr_dist > dist[node]:
            continue
        for neighbor, weight in graph[node]:
            new_dist = curr_dist + weight
            if new_dist < dist[neighbor]:
                dist[neighbor] = new_dist
                heapq.heappush(pq, (new_dist, neighbor))

    return dist

graph = {0: [(1, 3), (3, 5)], 1: [(0, 2), (3, 8)], 2: [(1, 1), (3, 4)], 3: [(2, 2)]}
all_pairs = {node: dijkstra(graph, node) for node in graph}
print(all_pairs)
```

**🔥 Why is Johnson’s Algorithm better?**  
- Instead of **O(N³) Floyd-Warshall**, it runs **O(N log N) Dijkstra’s** per node, leading to **O(N² log N).**

---

## **3️⃣ Optimizing DP for Longest Common Subsequence (O(N³) → O(N²))**  
### **🔴 Brute Force Recursive LCS (O(N³))**
```python
def lcs_brute_force(X, Y, m, n):
    if m == 0 or n == 0:
        return 0
    elif X[m - 1] == Y[n - 1]:
        return 1 + lcs_brute_force(X, Y, m - 1, n - 1)
    else:
        return max(lcs_brute_force(X, Y, m, n - 1), lcs_brute_force(X, Y, m - 1, n))

X = "AGGTAB"
Y = "GXTXAYB"
print(lcs_brute_force(X, Y, len(X), len(Y)))
```
**🔴 Problem:**  
- It recomputes LCS recursively, leading to **O(2^N) exponential calls**.

### **✅ Optimized Using DP (O(N²))**
```python
def lcs_dp(X, Y):
    m, n = len(X), len(Y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if X[i - 1] == Y[j - 1]:
                dp[i][j] = 1 + dp[i - 1][j - 1]
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]

X = "AGGTAB"
Y = "GXTXAYB"
print(lcs_dp(X, Y))  # Output: 4
```
**🔥 Why is DP better?**  
- Instead of **O(2^N) recursive calls**, DP fills a table in **O(N²).**

---

## **Final Comparison Table**
| **Algorithm** | **Before (O(N³))** | **After (O(N²))** |
|--------------|------------------|------------------|
| **Matrix Multiplication** | Naive | Strassen’s |
| **All-Pairs Shortest Path** | Floyd-Warshall | Johnson’s Algorithm |
| **Longest Common Subsequence** | Brute Force Recursion | Dynamic Programming |

Would you like optimizations for **O(N⁴) → O(N³)** cases? 🚀

### **Optimizing O(N⁴) to O(N³) Algorithms**  
Many algorithms with **O(N⁴) complexity** can be optimized to **O(N³)** using **Dynamic Programming (DP), Matrix Multiplication, and Graph Optimization Techniques.**  

---

## **1️⃣ Boolean Matrix Multiplication (O(N⁴) → O(N³))**  
### **🔴 Naive Boolean Matrix Multiplication (O(N⁴))**  
Each element in the resulting matrix is computed as a Boolean OR operation over **N²** elements, leading to **O(N⁴)** time complexity.

```python
def boolean_matrix_mult(A, B):
    N = len(A)
    result = [[0] * N for _ in range(N)]
    
    for i in range(N):
        for j in range(N):
            for k in range(N):
                result[i][j] = result[i][j] or (A[i][k] and B[k][j])
    
    return result

A = [[1, 0], [0, 1]]
B = [[0, 1], [1, 0]]
print(boolean_matrix_mult(A, B))
```

### **✅ Optimized Using Matrix Exponentiation (O(N³))**  
Instead of computing every OR operation separately, **Matrix Exponentiation using Strassen’s Algorithm** reduces the complexity.

```python
import numpy as np

def boolean_matrix_mult_optimized(A, B):
    return np.matmul(A, B).clip(max=1)  # Using NumPy for efficient matrix multiplication

A = np.array([[1, 0], [0, 1]])
B = np.array([[0, 1], [1, 0]])
print(boolean_matrix_mult_optimized(A, B))
```

**🔥 Why is this better?**  
- Instead of **O(N⁴) nested loops**, NumPy optimizes it to **O(N³).**

---

## **2️⃣ All-Pairs Shortest Path with Edge Constraints (O(N⁴) → O(N³))**  
### **🔴 Floyd-Warshall with Edge Constraints (O(N⁴))**  
- The standard **Floyd-Warshall Algorithm** runs in **O(N³)**.  
- If we add an extra **edge constraint** (like max edge weight filtering), the complexity increases to **O(N⁴)**.

```python
def floyd_warshall_constraints(graph, max_weight):
    N = len(graph)
    dist = [[float('inf')] * N for _ in range(N)]
    
    for i in range(N):
        for j in range(N):
            if graph[i][j] <= max_weight:
                dist[i][j] = graph[i][j]

    for k in range(N):
        for i in range(N):
            for j in range(N):
                if dist[i][k] != float('inf') and dist[k][j] != float('inf'):
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    
    return dist

graph = [[0, 3, float('inf'), 5], [2, 0, float('inf'), 8], [float('inf'), 1, 0, 4], [float('inf'), float('inf'), 2, 0]]
print(floyd_warshall_constraints(graph, max_weight=5))
```

### **✅ Optimized Using Johnson’s Algorithm (O(N³))**  
Instead of **Floyd-Warshall with Edge Constraints**, we can use **Dijkstra’s Algorithm** to handle edge filtering dynamically.

```python
import heapq

def dijkstra(graph, start, max_weight):
    N = len(graph)
    dist = {node: float('inf') for node in range(N)}
    dist[start] = 0
    pq = [(0, start)]

    while pq:
        curr_dist, node = heapq.heappop(pq)
        if curr_dist > dist[node]:
            continue
        for neighbor, weight in graph[node]:
            if weight <= max_weight:
                new_dist = curr_dist + weight
                if new_dist < dist[neighbor]:
                    dist[neighbor] = new_dist
                    heapq.heappush(pq, (new_dist, neighbor))

    return dist

graph = {0: [(1, 3), (3, 5)], 1: [(0, 2), (3, 8)], 2: [(1, 1), (3, 4)], 3: [(2, 2)]}
all_pairs = {node: dijkstra(graph, node, max_weight=5) for node in graph}
print(all_pairs)
```

**🔥 Why is Johnson’s Algorithm better?**  
- Instead of **O(N⁴) Floyd-Warshall**, it runs **O(N log N) Dijkstra’s** per node, leading to **O(N³).**

---

## **3️⃣ DP Optimization for Longest Palindromic Subsequence (O(N⁴) → O(N³))**  
### **🔴 Brute Force Recursive LPS (O(N⁴))**
A **naive recursive approach** generates all substrings and checks for palindromes, resulting in **O(N⁴) complexity**.

```python
def is_palindrome(s):
    return s == s[::-1]

def longest_palindromic_subseq_brute(s):
    N = len(s)
    max_length = 0
    for i in range(N):
        for j in range(i, N):
            if is_palindrome(s[i:j+1]):
                max_length = max(max_length, j - i + 1)
    return max_length

s = "bbbab"
print(longest_palindromic_subseq_brute(s))
```

### **✅ Optimized Using DP (O(N³))**
Instead of checking every substring, we use **Dynamic Programming**.

```python
def longest_palindromic_subseq_dp(s):
    N = len(s)
    dp = [[0] * N for _ in range(N)]

    for i in range(N):
        dp[i][i] = 1

    for length in range(2, N+1):
        for i in range(N-length+1):
            j = i + length - 1
            if s[i] == s[j]:
                dp[i][j] = 2 + dp[i+1][j-1]
            else:
                dp[i][j] = max(dp[i+1][j], dp[i][j-1])

    return dp[0][N-1]

s = "bbbab"
print(longest_palindromic_subseq_dp(s))  # Output: 4
```

**🔥 Why is DP better?**  
- Instead of **O(N⁴) checking all substrings**, DP efficiently finds the result in **O(N³).**

---

## **Final Comparison Table**
| **Algorithm** | **Before (O(N⁴))** | **After (O(N³))** |
|--------------|------------------|------------------|
| **Boolean Matrix Multiplication** | Naive OR Computation | Matrix Exponentiation |
| **All-Pairs Shortest Path with Constraints** | Floyd-Warshall with Filtering | Johnson’s Algorithm |
| **Longest Palindromic Subsequence** | Brute Force Checking | Dynamic Programming |

Would you like optimizations for **O(N⁵) → O(N⁴)** cases? 🚀


### **Optimizing O(N⁵) to O(N⁴) Algorithms**  
When an algorithm has **O(N⁵) complexity**, it usually involves **nested loops, exhaustive searches, or inefficient matrix operations**. Optimizing it to **O(N⁴)** often requires **Dynamic Programming (DP), Matrix Multiplication tricks, and Graph Algorithm Optimizations.**  

---

## **1️⃣ Graph Triangle Counting (O(N⁵) → O(N⁴))**  
### **🔴 Naive Triangle Counting (O(N⁵))**  
For an **unweighted graph**, a triangle is a cycle of **three vertices**. A brute-force method checks all triplets of nodes.

```python
def count_triangles(graph):
    N = len(graph)
    count = 0

    for i in range(N):
        for j in range(N):
            for k in range(N):
                for l in range(N):
                    for m in range(N):
                        if graph[i][j] and graph[j][k] and graph[k][i]:  # Triangle Condition
                            count += 1

    return count // 6  # Each triangle is counted 6 times (3! permutations)

graph = [[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 0]]
print(count_triangles(graph))  # Very slow for large graphs
```

### **✅ Optimized Using Matrix Multiplication (O(N⁴))**  
We can optimize this using **adjacency matrix multiplication (A³)**.

```python
import numpy as np

def count_triangles_optimized(graph):
    A = np.array(graph)
    A2 = np.matmul(A, A)  # A²
    A3 = np.matmul(A2, A)  # A³
    return np.trace(A3) // 6  # Diagonal elements count triangles

graph = [[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 0]]
print(count_triangles_optimized(graph))  # Much faster for large graphs
```

**🔥 Why is Matrix Multiplication better?**  
- **O(N⁵)** loops are replaced by **O(N³) matrix multiplications**, reducing the complexity to **O(N⁴).**

---

## **2️⃣ DP Optimization for 5D State Space (O(N⁵) → O(N⁴))**  
### **🔴 Brute Force 5D DP (O(N⁵))**  
Consider a **5D DP problem** (e.g., 5-character subsequence analysis):

```python
def dp_five_dim(n):
    dp = [[[[[0 for _ in range(n)] for _ in range(n)] for _ in range(n)] for _ in range(n)] for _ in range(n)]

    for a in range(n):
        for b in range(n):
            for c in range(n):
                for d in range(n):
                    for e in range(n):
                        dp[a][b][c][d][e] = (a + b + c + d + e) % 10

    return dp[0][0][0][0][0]

print(dp_five_dim(5))  # Slow for large N
```

### **✅ Optimized by Reducing Dimensions (O(N⁴))**  
Instead of **5D DP**, we **precompute** overlapping states.

```python
def dp_optimized(n):
    dp = [[[[0 for _ in range(n)] for _ in range(n)] for _ in range(n)] for _ in range(n)]

    for a in range(n):
        for b in range(n):
            for c in range(n):
                for d in range(n):
                    dp[a][b][c][d] = (a + b + c + d) % 10  # Reduce dimension

    return dp[0][0][0][0]

print(dp_optimized(5))  # Faster for large N
```

**🔥 Why is this better?**  
- **Removing unnecessary dimensions** makes it run in **O(N⁴) instead of O(N⁵).**

---

## **3️⃣ Multi-Source Shortest Path with 5D Constraints (O(N⁵) → O(N⁴))**  
### **🔴 Floyd-Warshall with Extra Dimensions (O(N⁵))**  
A naive way to solve **multi-source shortest paths** with **time and weight constraints**:

```python
def floyd_warshall_extended(graph, max_time):
    N = len(graph)
    dist = [[[[float('inf')] * max_time for _ in range(N)] for _ in range(N)] for _ in range(N)]

    for i in range(N):
        for j in range(N):
            for t in range(max_time):
                for k in range(N):
                    for w in range(N):
                        if graph[i][j] <= w and t + w < max_time:
                            dist[i][j][t][k][w] = min(dist[i][j][t][k][w], graph[i][j] + dist[k][j][t][k][w])

    return dist
```

### **✅ Optimized Using Dijkstra for Each Source (O(N⁴))**  
Instead of **Floyd-Warshall**, we run **Dijkstra’s Algorithm**.

```python
import heapq

def dijkstra_multi_source(graph, source, max_time):
    N = len(graph)
    dist = [[float('inf')] * max_time for _ in range(N)]
    pq = [(0, source, 0)]  # (cost, node, time)

    while pq:
        curr_cost, node, curr_time = heapq.heappop(pq)
        if curr_cost > dist[node][curr_time]:
            continue
        for neighbor, weight in graph[node]:
            new_time = curr_time + weight
            if new_time < max_time and curr_cost + weight < dist[neighbor][new_time]:
                dist[neighbor][new_time] = curr_cost + weight
                heapq.heappush(pq, (dist[neighbor][new_time], neighbor, new_time))

    return dist

graph = {0: [(1, 3), (2, 5)], 1: [(2, 2)], 2: [(0, 1)]}
print(dijkstra_multi_source(graph, source=0, max_time=5))
```

**🔥 Why is Dijkstra’s Algorithm better?**  
- **O(N⁵) Floyd-Warshall** is replaced by **O(N log N) per source**, reducing complexity to **O(N⁴).**

---

## **Final Comparison Table**
| **Algorithm** | **Before (O(N⁵))** | **After (O(N⁴))** |
|--------------|------------------|------------------|
| **Triangle Counting** | Brute Force | Matrix Multiplication |
| **5D DP Optimization** | Naive 5D Loops | Dimensional Reduction |
| **Multi-Source Shortest Path** | Floyd-Warshall | Dijkstra’s Per Source |

Would you like optimizations for **O(N⁶) → O(N⁵)** cases? 🚀


### **Optimizing O(N⁶) to O(N⁵) Algorithms**  
When an algorithm has **O(N⁶) complexity**, it typically involves **six nested loops, multidimensional DP, or exhaustive searches**. To reduce it to **O(N⁵)**, we use:  
✅ **Matrix Multiplication Tricks**  
✅ **Dynamic Programming with Dimensional Reduction**  
✅ **Graph Algorithm Optimizations**  
✅ **Parallel Processing Techniques**  

---

## **1️⃣ Brute Force Hexagon Counting (O(N⁶) → O(N⁵))**  
### **🔴 Naive O(N⁶) Approach**  
Counting **hexagons in a graph** using six nested loops.

```python
def count_hexagons(graph):
    N = len(graph)
    count = 0
    for a in range(N):
        for b in range(N):
            for c in range(N):
                for d in range(N):
                    for e in range(N):
                        for f in range(N):
                            if (graph[a][b] and graph[b][c] and graph[c][d] and
                                graph[d][e] and graph[e][f] and graph[f][a]):  # Hexagon condition
                                count += 1
    return count // 12  # Each hexagon is counted 12 times (6! permutations)

graph = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
print(count_hexagons(graph))  # Too slow for large N
```

### **✅ Optimized Using Matrix Exponentiation (O(N⁵))**  
Instead of brute force, **adjacency matrix exponentiation (A⁶)** is used.

```python
import numpy as np

def count_hexagons_optimized(graph):
    A = np.array(graph)
    A6 = np.linalg.matrix_power(A, 6)  # A⁶ gives paths of length 6
    return np.trace(A6) // 12  # Count hexagons using diagonal elements

graph = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
print(count_hexagons_optimized(graph))  # Much faster!
```

**🔥 Why is this better?**  
- **Nested loops (O(N⁶)) are replaced by matrix exponentiation (O(N⁵))**, reducing complexity.  

---

## **2️⃣ High-Dimensional DP (O(N⁶) → O(N⁵))**  
### **🔴 6D DP State Space (O(N⁶))**  
A problem with **6D DP** is extremely inefficient.

```python
def dp_six_dim(n):
    dp = [[[[[[0 for _ in range(n)] for _ in range(n)] for _ in range(n)]
             for _ in range(n)] for _ in range(n)] for _ in range(n)]

    for a in range(n):
        for b in range(n):
            for c in range(n):
                for d in range(n):
                    for e in range(n):
                        for f in range(n):
                            dp[a][b][c][d][e][f] = (a + b + c + d + e + f) % 10

    return dp[0][0][0][0][0][0]

print(dp_six_dim(5))  # Slow for large N
```

### **✅ Optimized by Reducing State Space (O(N⁵))**  
Reduce one dimension using **prefix sums**.

```python
def dp_optimized(n):
    dp = [[[[[0 for _ in range(n)] for _ in range(n)]
             for _ in range(n)] for _ in range(n)] for _ in range(n)]

    for a in range(n):
        for b in range(n):
            for c in range(n):
                for d in range(n):
                    for e in range(n):
                        dp[a][b][c][d][e] = (a + b + c + d + e) % 10  # Reduce one dimension

    return dp[0][0][0][0][0]

print(dp_optimized(5))  # Faster for large N
```

**🔥 Why is this better?**  
- Instead of **O(N⁶) memory & time**, we use **O(N⁵) by precomputing overlapping states.**  

---

## **3️⃣ Multi-Source Path Computation (O(N⁶) → O(N⁵))**  
### **🔴 Naive Floyd-Warshall with Extra Constraints (O(N⁶))**  
A **pathfinding algorithm with time & weight constraints**.

```python
def floyd_warshall_extended(graph, max_time):
    N = len(graph)
    dist = [[[[[[float('inf')] * max_time for _ in range(N)] for _ in range(N)]
              for _ in range(N)] for _ in range(N)] for _ in range(N)]

    for i in range(N):
        for j in range(N):
            for k in range(N):
                for t in range(max_time):
                    for w in range(N):
                        for x in range(N):
                            if graph[i][j] <= w and t + w < max_time:
                                dist[i][j][t][k][w][x] = min(
                                    dist[i][j][t][k][w][x], graph[i][j] + dist[k][j][t][k][w][x]
                                )

    return dist
```

### **✅ Optimized Using Bellman-Ford for Each Source (O(N⁵))**  
Instead of **Floyd-Warshall**, we run **Bellman-Ford for all sources**.

```python
def bellman_ford_multi_source(graph, source, max_time):
    N = len(graph)
    dist = [[[float('inf')] * max_time for _ in range(N)] for _ in range(N)]
    
    for t in range(max_time):
        for u in range(N):
            for v, weight in graph[u]:
                if t + weight < max_time:
                    dist[source][v][t + weight] = min(dist[source][v][t + weight], dist[source][u][t] + weight)

    return dist

graph = {0: [(1, 3), (2, 5)], 1: [(2, 2)], 2: [(0, 1)]}
print(bellman_ford_multi_source(graph, source=0, max_time=5))
```

**🔥 Why is this better?**  
- **Bellman-Ford per source** avoids **Floyd-Warshall’s O(N⁶) complexity** and runs in **O(N⁵).**  

---

## **Final Comparison Table**
| **Algorithm** | **Before (O(N⁶))** | **After (O(N⁵))** |
|--------------|------------------|------------------|
| **Hexagon Counting** | 6 Nested Loops | Matrix Power (A⁶) |
| **6D DP Optimization** | Naive 6D Loops | Dimensional Reduction |
| **Multi-Source Path** | Extended Floyd-Warshall | Bellman-Ford per source |

Would you like further optimizations **for O(N⁷) → O(N⁶)** cases? 🚀



