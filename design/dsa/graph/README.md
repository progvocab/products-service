# Topics
- Undirected Graph 
- Depth First Search 
- Stack Data structure 
- Recursion 
- Breadth-First Search 
- Queue
- Shortest Part from a Vertex 
- Greedy Algorithm 
- Dijkstra 
- Priority Queue 
- Tuples 
- Travelling Salesman Problem 
- Dynamic Programming 
- Bit Manipulation 
- LRU Cache
- Detecting Cycles 
- Disjoint Set
- Union-Find Datastructures 
- Minimum Spanning Tree 
- Negative Edge
- Bellman Ford
- Bridge 
- Articulation Points 



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

### ✅ 4. **Cycle Detection (Directed Graph, DFS-based)**

```python
def has_cycle(graph):
    visited, rec = set(), set()
    def dfs(v):
        if v in rec: return True
        if v in visited: return False
        visited.add(v); rec.add(v)
        if any(dfs(nei) for nei in graph.get(v, [])): return True
        rec.remove(v); return False
    return any(dfs(v) for v in graph)
```

---

### ✅ 5. **Topological Sort (Kahn’s Algorithm)**

```python
from collections import deque, defaultdict
def topo_sort(graph):
    indegree = defaultdict(int)
    for u in graph: 
        for v in graph[u]: indegree[v] += 1
    q = deque([u for u in graph if indegree[u]==0])
    res = []
    while q:
        u = q.popleft(); res.append(u)
        for v in graph[u]:
            indegree[v] -= 1
            if indegree[v] == 0: q.append(v)
    return res
```

---

### ✅ 7. **Detect Strongly Connected Components (Tarjan’s Algorithm)**

```python
def tarjans_scc(graph):
    idx, stack, indices, lowlink, on_stack, sccs = 0, [], {}, {}, set(), []
    def strongconnect(v):
        nonlocal idx
        indices[v] = lowlink[v] = idx; idx += 1
        stack.append(v); on_stack.add(v)
        for w in graph.get(v, []):
            if w not in indices:
                strongconnect(w); lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], indices[w])
        if lowlink[v] == indices[v]:
            scc, w = [], None
            while w != v:
                w = stack.pop(); on_stack.remove(w); scc.append(w)
            sccs.append(scc)
    for v in graph:
        if v not in indices: strongconnect(v)
    return sccs
```

---

### ✅ 8. **Check if Graph is Bipartite**

```python
def is_bipartite(graph):
    color = {}
    def dfs(u, c):
        color[u] = c
        return all(color.get(v, c^1) == c^1 and dfs(v, c^1) for v in graph[u] if v not in color or color[v] == c^1)
    return all(v in color or dfs(v, 0) for v in graph)
```

---

Here's a **concise set of Python implementations** for advanced graph problems:

---

### ✅ 1. **Maximum Flow (Ford-Fulkerson using DFS)**

```python
def ford_fulkerson(capacity, source, sink):
    n = len(capacity)
    flow = [[0]*n for _ in range(n)]

    def dfs(u, t, f, visited):
        if u == t: return f
        visited[u] = True
        for v in range(n):
            if not visited[v] and capacity[u][v] - flow[u][v] > 0:
                df = dfs(v, t, min(f, capacity[u][v] - flow[u][v]), visited)
                if df > 0:
                    flow[u][v] += df
                    flow[v][u] -= df
                    return df
        return 0

    max_flow = 0
    while True:
        visited = [False]*n
        pushed = dfs(source, sink, float('inf'), visited)
        if not pushed: break
        max_flow += pushed
    return max_flow
```

> **Usage:**

```python
# Node 0 -> 1(10), 0 -> 2(5), etc.
capacity = [
    [0, 10, 5, 0],
    [0, 0, 15, 10],
    [0, 0, 0, 10],
    [0, 0, 0, 0]
]
print(ford_fulkerson(capacity, 0, 3))  # Output: 15
```

---

### ✅ 2. **Planarity Test (Using `networkx`)**

```python
import networkx as nx

def is_planar(graph_edges):
    G = nx.Graph()
    G.add_edges_from(graph_edges)
    return nx.check_planarity(G)[0]
```

> **Usage:**

```python
edges = [(0,1),(1,2),(2,3),(3,4),(4,0),(0,2)]  # K5 is not planar
print(is_planar(edges))  # False
```

---

### ✅ 3. **Maximum Bipartite Matching (Hungarian Algorithm via `networkx`)**

```python
import networkx as nx

def max_bipartite_matching(edges, U, V):
    G = nx.Graph()
    G.add_nodes_from(U, bipartite=0)
    G.add_nodes_from(V, bipartite=1)
    G.add_edges_from(edges)
    return nx.bipartite.maximum_matching(G)
```

> **Usage:**

```python
U, V = [1,2], ['a','b']
edges = [(1, 'a'), (2, 'a'), (2, 'b')]
print(max_bipartite_matching(edges, U, V))  # {1: 'a', 'a': 1, 2: 'b', 'b': 2}
```

---

### ✅ 4. **Travelling Salesman Problem (TSP, brute force)**

```python
from itertools import permutations

def tsp_brute_force(dist):
    n, min_cost = len(dist), float('inf')
    for perm in permutations(range(1, n)):
        cost = dist[0][perm[0]] + sum(dist[perm[i]][perm[i+1]] for i in range(n-2)) + dist[perm[-1]][0]
        min_cost = min(min_cost, cost)
    return min_cost
```

> **Usage:**

```python
dist = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
]
print(tsp_brute_force(dist))  # 80
```

---

The **Union-Find** data structure (also called **Disjoint Set Union (DSU)**) is widely used in graph algorithms and systems where you need to efficiently manage **groupings of elements** and **check if two elements are in the same group**.

---

## ✅ **Real-World and Algorithmic Use Cases of Union-Find**

| Use Case                                        | Description                                                                                           |
| ----------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| **1. Kruskal’s Algorithm**                      | To build a Minimum Spanning Tree (MST), we use Union-Find to check if adding an edge creates a cycle. |
| **2. Connected Components**                     | Efficiently group nodes in a graph into components.                                                   |
| **3. Cycle Detection in Undirected Graph**      | Detect if adding an edge connects nodes already in the same set (cycle).                              |
| **4. Network Connectivity**                     | Check whether two users/computers/devices are connected.                                              |
| **5. Image Processing**                         | Identify connected pixel regions (blob detection, flood fill).                                        |
| **6. Social Network Friend Circles**            | Merge users into "friend groups" if they are connected.                                               |
| **7. Dynamic Connectivity**                     | Maintain changing network connectivity (e.g., adding/removing edges in real-time).                    |
| **8. Percolation Theory / Physics Simulations** | Determine when systems become "connected" from top to bottom.                                         |
| **9. Compiler Design (Type Inference)**         | To merge equivalent types.                                                                            |
| **10. Version Control Internals**               | Merging file histories or branches.                                                                   |

---

## 🔧 Python Example: Cycle Detection in Undirected Graph using Union-Find

```python
def find(parent, x):
    if parent[x] != x:
        parent[x] = find(parent, parent[x])  # Path compression
    return parent[x]

def union(parent, rank, x, y):
    rootX, rootY = find(parent, x), find(parent, y)
    if rootX == rootY:
        return False  # Cycle detected
    if rank[rootX] < rank[rootY]:
        parent[rootX] = rootY
    else:
        parent[rootY] = rootX
        if rank[rootX] == rank[rootY]:
            rank[rootX] += 1
    return True

def has_cycle(edges, n):
    parent = list(range(n))
    rank = [0]*n
    for u, v in edges:
        if not union(parent, rank, u, v):
            return True
    return False
```

> **Usage:**

```python
edges = [(0,1),(1,2),(2,0)]
print(has_cycle(edges, 3))  # True
```

---

Would you like an advanced implementation with **union by size**, or **parallel DSU** for huge datasets?





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


Here are **key algorithms related to bridges and articulation points** in graph theory, along with **related concepts and algorithms** that solve similar or interconnected problems:

---

### 🔹 1. **Tarjan’s Algorithm**

* **Purpose:** Finds **articulation points**, **bridges**, and **strongly connected components (SCC)**.
* **Approach:** DFS-based, tracks `discovery time` and `low time` for each node.
* **Time Complexity:** `O(V + E)`
* **Variants:**

  * **Tarjan for articulation points and bridges**
  * **Tarjan for strongly connected components (SCCs)** (used in directed graphs)

---

### 🔹 2. **Kosaraju’s Algorithm**

* **Purpose:** Finds **strongly connected components** (SCC) in a **directed graph**.
* **Approach:**

  1. Do DFS and push nodes onto a stack by finish time.
  2. Reverse the graph.
  3. Do DFS in order of stack to find SCCs.
* **Time Complexity:** `O(V + E)`

---

### 🔹 3. **Gabow’s Algorithm**

* **Purpose:** Also used for finding **strongly connected components**.
* **Improved stack-based version** of Tarjan’s for SCCs.
* **Time Complexity:** `O(V + E)`

---

### 🔹 4. **Union-Find / Disjoint Set Union (DSU)**

* **Purpose:** Detect **connected components**, especially useful in **offline bridge-finding algorithms** or **Kruskal’s MST**.
* **Time Complexity:** Almost `O(1)` per operation with path compression.

---

### 🔹 5. **Low-Link Values Algorithm (DFS-based)**

* **Purpose:** Foundational concept used in **Tarjan’s** and **bridge-finding algorithms**.
* Maintains:

  * `disc[v]`: discovery time
  * `low[v]`: lowest reachable ancestor
* Helps detect:

  * **Back edges** (which indicate cycles)
  * **Bridges** and **articulation points**

---

### 🔹 6. **Depth-First Search (DFS) Tree Concepts**

Used to derive:

* **DFS tree edges**
* **Back edges**
* **Forward edges**
* **Cross edges**

Important for identifying **cycles**, **bridges**, and **cut-vertices**.

---

### 🔹 7. **Biconnected Components**

* **Definition:** Maximal subgraphs where the removal of any single vertex does **not disconnect** the graph.
* **Related to articulation points**.
* Can be found using DFS + stack.

---

### 🔹 8. **Bridge Tree / Block Tree**

* **Tree formed by collapsing biconnected components** into single nodes.
* Used in network reliability, optimization of routes, and fault tolerance.

---

## 🔸 Related Concepts / Problems

| Concept                    | Related Algorithms                      |
| -------------------------- | --------------------------------------- |
| **Cycle Detection**        | DFS, Union-Find (for undirected graphs) |
| **Connected Components**   | BFS/DFS, Union-Find                     |
| **Articulation Points**    | Tarjan’s, DFS low-link                  |
| **Bridges (Cut-edges)**    | Tarjan’s, DFS low-link                  |
| **SCCs (Directed Graph)**  | Kosaraju’s, Tarjan’s, Gabow’s           |
| **Biconnected Components** | DFS + stack                             |
| **Minimum Spanning Tree**  | Kruskal’s (can help detect bridges)     |

---

### 📌 Visual Summary

```
       Graph Problem
          ├── Cycle Detection (DFS / Union-Find)
          ├── Bridges ←────┐
          ├── Articulation Points ←─── Tarjan’s Algorithm
          ├── Strongly Connected Components ←── Kosaraju / Tarjan
          └── Biconnected Components ←── DFS + Stack
```

---

Would you like a visual diagram, code for these related algorithms, or comparisons between them in a table?

Here’s the clean comparison — short, clear, but deep enough so you won’t mix them up again.

---

## 1. **Basic Idea**

| Feature                  | **Tree**                                                           | **Acyclic Graph** (DAG)                                                                                                  |
| ------------------------ | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------ |
| **Type**                 | Special kind of acyclic graph                                      | More general — can include trees as a subset                                                                             |
| **Cycles**               | No cycles                                                          | No cycles                                                                                                                |
| **Connectedness**        | Always connected                                                   | May be disconnected                                                                                                      |
| **Edges**                | Exactly `n - 1` edges for `n` vertices                             | Can have fewer or more (up to `n(n-1)/2` in undirected case without cycles)                                              |
| **Parent-child concept** | Always has one root, every node except root has exactly one parent | No strict parent-child rule — a node can have multiple incoming edges                                                    |
| **Direction**            | Can be undirected or rooted (directed edges from parent to child)  | Typically discussed as **directed** (Directed Acyclic Graph = DAG), but undirected acyclic graphs are called **forests** |
| **Hierarchy**            | Strict hierarchy                                                   | Not necessarily hierarchical, can be a partial order                                                                     |

---

## 2. **Visual Difference**

**Tree** (connected, n-1 edges)

```
    A
   / \
  B   C
     / \
    D   E
```

**DAG** (acyclic but can have multiple parents and cross-links)

```
A → B
A → C
B → D
C → D
```

Here `D` has two parents (`B` and `C`), so it’s **not** a tree.

---

## 3. **Key Rules to Remember**

* **Every tree is an acyclic graph, but not every acyclic graph is a tree.**
* If an **acyclic graph** is **connected** and has exactly **n − 1 edges**, it’s a **tree**.
* If an acyclic graph is disconnected, it’s called a **forest** (a collection of trees).

---

If you want, I can also give you **a quick table of formulas** for edges, degrees, and properties so you can identify in O(1) whether a given acyclic graph is a tree. That’s a neat trick for interviews.
