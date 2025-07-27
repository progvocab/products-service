## **Graph in Data Structures & Algorithms (DSA)**  

A **graph** is a non-linear data structure consisting of **nodes (vertices)** connected by **edges**. Graphs are widely used in real-world applications such as **social networks, shortest path algorithms (Google Maps), recommendation systems, and network topology**.

---

## **📌 Graph Concepts**
1. **Vertex (Node)** - A point in the graph.
2. **Edge** - A connection between two vertices.
3. **Weight** (Optional) - A value assigned to an edge in weighted graphs.
4. **Directed Graph** - Edges have direction (`A → B` but not `B → A`).
5. **Undirected Graph** - Edges have no direction (`A ↔ B`).
6. **Adjacency List** - A list where each node stores its neighbors.
7. **Adjacency Matrix** - A 2D array representing connections between nodes.
8. **Connected Graph** - A graph where every vertex is reachable.
9. **Cyclic & Acyclic Graph** - A cyclic graph contains cycles; an acyclic one does not.
10. **Complete Graph** - A complete graph is a simple undirected graph in which every pair of distinct vertices is connected by a unique edge,  a complete graph with n vertices has n(n - 1)/2 edges.
11. A bipartite graph divides its vertices into two disjoint sets such that every edge connects a vertex from one set to the other.
12. A complete bipartite graph connects every vertex in set A (size m) to every vertex in set B (size n).
13. Planar graphs 
---

## **📌 Time & Space Complexity**
| Operation | Adjacency List | Adjacency Matrix |
|-----------|---------------|------------------|
| **Storage** | `O(V + E)` | `O(V²)` |
| **Add Edge** | `O(1)` | `O(1)` |
| **Remove Edge** | `O(E)` | `O(1)` |
| **Check Edge** | `O(E)` | `O(1)` |
| **Iterate Neighbors** | `O(V + E)` | `O(V)` |

📌 **V = Number of vertices, E = Number of edges**

- **Adjacency List is better for sparse graphs (`E << V²`)**  
- **Adjacency Matrix is better for dense graphs (`E ≈ V²`)**

---

## **📌 Graph Representation in Python**
### **1️⃣ Adjacency List (Efficient for Sparse Graphs)**
```python
class Graph:
    def __init__(self):
        self.graph = {}

    def add_edge(self, u, v):
        if u not in self.graph:
            self.graph[u] = []
        self.graph[u].append(v)

    def display(self):
        for node in self.graph:
            print(f"{node} -> {self.graph[node]}")

g = Graph()
g.add_edge(1, 2)
g.add_edge(1, 3)
g.add_edge(2, 4)
g.display()
```
**Output:**
```
1 -> [2, 3]
2 -> [4]
```

---

### **2️⃣ Adjacency Matrix (Efficient for Dense Graphs)**
```python
class GraphMatrix:
    def __init__(self, size):
        self.size = size
        self.matrix = [[0] * size for _ in range(size)]

    def add_edge(self, u, v):
        self.matrix[u][v] = 1  # For directed graph
        self.matrix[v][u] = 1  # Add this line for an undirected graph

    def display(self):
        for row in self.matrix:
            print(row)

g = GraphMatrix(4)
g.add_edge(0, 1)
g.add_edge(1, 2)
g.add_edge(2, 3)
g.display()
```
**Output:**
```
[0, 1, 0, 0]
[1, 0, 1, 0]
[0, 1, 0, 1]
[0, 0, 1, 0]
```

---

## **📌 Graph Traversal Algorithms**
### **1️⃣ Breadth-First Search (BFS)**
BFS explores level by level using a **queue** (FIFO).

```python
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])

    while queue:
        node = queue.popleft()
        if node not in visited:
            print(node, end=" ")
            visited.add(node)
            queue.extend(graph.get(node, []))

graph = {1: [2, 3], 2: [4, 5], 3: [6], 4: [], 5: [], 6: []}
bfs(graph, 1)
```
**Output:** `1 2 3 4 5 6`

📌 **Time Complexity:** `O(V + E)`, **Space Complexity:** `O(V)`

---

### **2️⃣ Depth-First Search (DFS)**
DFS explores depth-first using a **stack** (recursion).

```python
def dfs(graph, node, visited=set()):
    if node not in visited:
        print(node, end=" ")
        visited.add(node)
        for neighbor in graph.get(node, []):
            dfs(graph, neighbor, visited)

graph = {1: [2, 3], 2: [4, 5], 3: [6], 4: [], 5: [], 6: []}
dfs(graph, 1)
```
**Output:** `1 2 4 5 3 6`

📌 **Time Complexity:** `O(V + E)`, **Space Complexity:** `O(V)`

---

## **📌 Shortest Path Algorithms**
### **1️⃣ Dijkstra’s Algorithm (Single Source Shortest Path)**
Best for **weighted graphs** (e.g., road networks).

```python
import heapq

def dijkstra(graph, start):
    pq = [(0, start)]  # (distance, node)
    distances = {node: float('inf') for node in graph}
    distances[start] = 0

    while pq:
        curr_dist, node = heapq.heappop(pq)
        for neighbor, weight in graph[node]:
            distance = curr_dist + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    
    return distances

graph = {1: [(2, 1), (3, 4)], 2: [(3, 2), (4, 5)], 3: [(4, 1)], 4: []}
print(dijkstra(graph, 1))
```
**Output:** `{1: 0, 2: 1, 3: 3, 4: 4}`

📌 **Time Complexity:** `O((V + E) log V)`

---

## **📌 Use Cases of Graphs**
| **Application** | **Graph Type** |
|---------------|---------------|
| **Google Maps (Shortest Path)** | Weighted Graph |
| **Social Networks (Friendship Suggestions)** | Undirected Graph |
| **Recommendation Systems (Netflix, Amazon)** | Bipartite Graph |
| **Web Crawling (Search Engines)** | Directed Graph |
| **Airline Route Optimization** | Weighted Graph |
| **Circuit Design & Dependencies** | Directed Acyclic Graph (DAG) |

---

## **📌 Summary**
1. **Graphs** model relationships (e.g., social networks, road maps).
2. **Representation:** Adjacency List (`O(V + E)`) vs Adjacency Matrix (`O(V²)`).
3. **Traversal:** BFS (`O(V + E)`) and DFS (`O(V + E)`).
4. **Shortest Path:** Dijkstra (`O((V + E) log V)`) for weighted graphs.
5. **Use Cases:** Networking, Recommendations, Maps, Web Crawling.

Would you like help with a **specific graph problem**? 🚀

The difference between **O(N)**, **O(V)**, and **O(V + E)** lies in what each notation represents in terms of input size:  

- **O(N)**: Generic notation that represents complexity depending on **N**, where **N** can be **vertices (V), edges (E), or any input size**.  
- **O(V)**: Complexity depends **only on the number of vertices (V)** in a graph.  
- **O(V + E)**: Complexity depends on both **vertices (V) and edges (E)**, common in **graph algorithms**.

---

### **🚀 Differences in Graph Algorithms**
| **Algorithm** | **Complexity** | **Why?** |
|--------------|--------------|---------|
| **Iterating all nodes (vertices)** | `O(V)` | Only touches vertices. |
| **Iterating all edges** | `O(E)` | Only touches edges. |
| **BFS / DFS** | `O(V + E)` | Visits every vertex (`V`) and every edge (`E`). |
| **Dijkstra (Priority Queue)** | `O((V + E) log V)` | Uses BFS + heap operations (`log V`). |
| **Adjacency List Storage** | `O(V + E)` | Stores each vertex and its edges. |
| **Adjacency Matrix Storage** | `O(V²)` | Requires `V × V` space. |

---

### **📌 Example: BFS Complexity Analysis**
```python
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])

    while queue:
        node = queue.popleft()
        if node not in visited:
            print(node, end=" ")
            visited.add(node)
            queue.extend(graph.get(node, []))

graph = {1: [2, 3], 2: [4, 5], 3: [6], 4: [], 5: [], 6: []}
bfs(graph, 1)
```
✅ **Complexity:** `O(V + E)`, because:  
- Each **vertex** is visited **once** → `O(V)`.  
- Each **edge** is processed **once** → `O(E)`.  

---

### **📌 Key Takeaways**
1. **O(V)** - Graph operations that depend **only on vertices** (e.g., iterating all nodes).  
2. **O(E)** - Graph operations that depend **only on edges** (e.g., counting edges).  
3. **O(V + E)** - Traversal algorithms that depend **on both vertices & edges** (e.g., BFS, DFS).  

Let me know if you need a real-world example! 🚀


[search](search.md)