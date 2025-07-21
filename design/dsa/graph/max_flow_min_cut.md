The **Max Flow** algorithm solves the problem of finding the **maximum possible flow** from a **source node** to a **sink node** in a **flow network**, where each edge has a capacity.

---

## 🧭 Real-World Applications

* Network routing (max bandwidth)
* Job assignment problems
* Bipartite matching
* Image segmentation
* Airline scheduling

---

## 🔹 Key Concepts

1. **Flow Network**: A directed graph where each edge has a capacity.
2. **Flow**: A value ≤ capacity that can be pushed through an edge.
3. **Source (s)**: Start node where flow originates.
4. **Sink (t)**: End node where flow should reach.
5. **Residual Graph**: Graph showing available capacity after current flow.

---

## 🔸 Ford–Fulkerson Algorithm

A popular method to compute **maximum flow** using **repeated DFS/BFS** to find *augmenting paths*.

**Steps**:

1. Initialize flow = 0.
2. While there is a path from `s` to `t` with available capacity:

   * Find the *minimum residual capacity* (bottleneck).
   * Add flow to the path.
   * Update residual capacities.
3. Repeat until no augmenting path exists.

**Time Complexity**:

* `O(E * max_flow)` (can be high if flow is large)

🔸 With **Edmonds-Karp** (BFS instead of DFS), the complexity becomes `O(VE²)`.


The **Ford-Fulkerson algorithm** is used to find the **maximum flow** in a flow network. It repeatedly finds augmenting paths from the **source to the sink** using **DFS** (or BFS), and updates the residual capacities of the edges.

---

### 💡 Key Concepts:

1. **Residual Graph**: For every edge `(u → v)`, if there's flow `f`, the residual capacity is `capacity - f`. Also, a reverse edge `(v → u)` is added with capacity `f`.

2. **Augmenting Path**: A path from source to sink in the residual graph where all edges have positive capacity.

3. **DFS for finding path**: We use DFS to search for augmenting paths.

---

### 🧠 Time Complexity:

Depends on the number of augmenting paths and the method used:

* Ford-Fulkerson with DFS: `O(max_flow × E)` where `E` is number of edges.

---

### ✅ Python Implementation using DFS:

```python
from collections import defaultdict

class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = defaultdict(dict)

    def add_edge(self, u, v, capacity):
        self.graph[u][v] = capacity
        if v not in self.graph or u not in self.graph[v]:
            self.graph[v][u] = 0  # Add reverse edge with 0 capacity

    def _dfs(self, s, t, visited, flow):
        visited.add(s)
        if s == t:
            return flow
        for neighbor in self.graph[s]:
            cap = self.graph[s][neighbor]
            if cap > 0 and neighbor not in visited:
                min_cap = min(flow, cap)
                result = self._dfs(neighbor, t, visited, min_cap)
                if result > 0:
                    self.graph[s][neighbor] -= result
                    self.graph[neighbor][s] += result
                    return result
        return 0

    def ford_fulkerson(self, source, sink):
        max_flow = 0
        while True:
            visited = set()
            flow = self._dfs(source, sink, visited, float('inf'))
            if flow == 0:
                break
            max_flow += flow
        return max_flow
```

---

### 🔍 Example:

```python
g = Graph(6)
g.add_edge(0, 1, 16)
g.add_edge(0, 2, 13)
g.add_edge(1, 2, 10)
g.add_edge(1, 3, 12)
g.add_edge(2, 1, 4)
g.add_edge(2, 4, 14)
g.add_edge(3, 2, 9)
g.add_edge(3, 5, 20)
g.add_edge(4, 3, 7)
g.add_edge(4, 5, 4)

source = 0
sink = 5
print("Max Flow:", g.ford_fulkerson(source, sink))
```

### Output:

```
Max Flow: 23
```

---

### 🛠️ Real-World Applications:

* Network bandwidth allocation
* Bipartite matching
* Job assignment problems
* Circulation with demands

Would you like a visualization or BFS-based Edmonds-Karp version?


---

## ✅ Python Example using Edmonds-Karp (BFS-based)

```python
from collections import deque, defaultdict

class MaxFlow:
    def __init__(self, graph):
        self.graph = graph
        self.residual = defaultdict(dict)
        for u in graph:
            for v in graph[u]:
                self.residual[u][v] = graph[u][v]
                self.residual[v][u] = 0  # reverse edge with 0 capacity

    def bfs(self, s, t, parent):
        visited = set()
        queue = deque([s])
        visited.add(s)
        while queue:
            u = queue.popleft()
            for v in self.residual[u]:
                if v not in visited and self.residual[u][v] > 0:
                    visited.add(v)
                    parent[v] = u
                    if v == t:
                        return True
                    queue.append(v)
        return False

    def edmonds_karp(self, s, t):
        max_flow = 0
        parent = {}
        while self.bfs(s, t, parent):
            # Find bottleneck
            path_flow = float('inf')
            v = t
            while v != s:
                u = parent[v]
                path_flow = min(path_flow, self.residual[u][v])
                v = u
            # Update residual capacities
            v = t
            while v != s:
                u = parent[v]
                self.residual[u][v] -= path_flow
                self.residual[v][u] += path_flow
                v = u
            max_flow += path_flow
        return max_flow
```

---

### 🔹 Example Usage

```python
graph = {
    's': {'a': 10, 'b': 5},
    'a': {'b': 15, 't': 10},
    'b': {'t': 10},
    't': {}
}

mf = MaxFlow(graph)
print("Max Flow:", mf.edmonds_karp('s', 't'))  # Output: 15
```

---

## 🧠 Visualization

```
      10       10
  s ------> a -----> t
   \         |
    \5       |15
     \       v
      ---> b -----> t
             10
```

---

## ✅ Summary

| Algorithm         | Strategy          | Time Complexity                    |
| ----------------- | ----------------- | ---------------------------------- |
| Ford-Fulkerson    | DFS               | O(E × max flow)                    |
| Edmonds-Karp      | BFS               | O(VE²)                             |
| Dinic’s Algorithm | Level Graph + DFS | O(V²E) or O(E√V) for unit capacity |

---

Let me know if you want:

* Dinic’s algorithm (faster for large graphs)
* Visualization tools
* Minimum cut from max flow (Max-Flow Min-Cut Theorem)


The **Ford-Fulkerson algorithm** (or its BFS variant, **Edmonds-Karp**) is not directly applicable to **weighted graphs like the one you've given** unless we interpret the weights as **capacities** in a **flow network**.

Let’s assume this matrix represents a **flow network**, and we want to calculate the **maximum flow from node 0 (source) to node 3 (sink)** using **Ford-Fulkerson**.

---

### 🔢 Step 1: Interpret the Adjacency Matrix

Given adjacency matrix:

```
       0     1     2     3
0   [ 0 ,   3 ,   4 ,  inf ]
1   [inf,   0 , inf ,   5  ]
2   [inf, inf,  0 ,   6  ]
3   [inf, inf, inf,   0  ]
```

Interpret as:

* Edge 0 → 1 with capacity 3
* Edge 0 → 2 with capacity 4
* Edge 1 → 3 with capacity 5
* Edge 2 → 3 with capacity 6

### 🧠 Step 2: Construct Residual Graph

We’ll use the Ford-Fulkerson method with DFS to find augmenting paths.

---

### 🔁 Step 3: Run Algorithm

**Residual Capacities**:

```
0 → 1: 3
0 → 2: 4
1 → 3: 5
2 → 3: 6
```

#### 🔍 First DFS Path:

* 0 → 1 → 3
* Min capacity along path = min(3, 5) = **3**
* Update residual graph:

  * 0→1 becomes 0 (3-3)
  * 1→3 becomes 2 (5-3)
  * Add reverse edges:

    * 1→0 += 3
    * 3→1 += 3
* **Flow added: 3**

#### 🔍 Second DFS Path:

* 0 → 2 → 3
* Min capacity = min(4, 6) = **4**
* Update residual graph:

  * 0→2 becomes 0 (4-4)
  * 2→3 becomes 2 (6-4)
  * Add reverse edges:

    * 2→0 += 4
    * 3→2 += 4
* **Flow added: 4**

#### 🚫 No more augmenting paths from 0 to 3:

* 0→1 = 0
* 0→2 = 0

---

### ✅ Final Max Flow:

Total flow = 3 (first path) + 4 (second path) = **7**

---

### 🧮 Summary Table of Residual Capacities:

| Edge  | Initial Capacity | Residual Capacity |
| ----- | ---------------- | ----------------- |
| 0 → 1 | 3                | 0                 |
| 0 → 2 | 4                | 0                 |
| 1 → 3 | 5                | 2                 |
| 2 → 3 | 6                | 2                 |
| 1 ← 0 | 0                | 3                 |
| 2 ← 0 | 0                | 4                 |
| 3 ← 1 | 0                | 3                 |
| 3 ← 2 | 0                | 4                 |

---

### 🟢 Final Answer: **Max Flow = 7**

Would you like the Python code for this exact matrix example?


To represent a **graph with capacities** (as needed for max-flow problems like Ford-Fulkerson), the most common representations are:

---

### ✅ 1. **Adjacency List with Capacity Dictionary**

Each node has a dictionary of connected nodes with their corresponding **capacity**.

Example from your matrix:

```python
graph = {
    0: {1: 3, 2: 4},
    1: {3: 5},
    2: {3: 6},
    3: {}
}
```

This means:

* Node `0` has an edge to `1` with capacity `3` and to `2` with capacity `4`.
* Node `1` has an edge to `3` with capacity `5`.
* Node `2` has an edge to `3` with capacity `6`.

---

### ✅ 2. **Adjacency Matrix (2D Array)**

You can also represent capacities using a 2D list (matrix). Replace `inf` with 0 to indicate "no edge" (as capacities cannot be infinite in max-flow problems).

#### Given matrix (original):

```
[ [ 0, 3, 4, inf ],
  [ inf, 0, inf, 5 ],
  [ inf, inf, 0, 6 ],
  [ inf, inf, inf, 0 ] ]
```

#### As a capacity matrix (converted to 0):

```python
capacity = [
    [0, 3, 4, 0],  # node 0
    [0, 0, 0, 5],  # node 1
    [0, 0, 0, 6],  # node 2
    [0, 0, 0, 0]   # node 3
]
```

This can be used in Ford-Fulkerson implementations.

---

### ✅ 3. **Edge List with Capacity**

You can also represent the graph as a list of `(from, to, capacity)` tuples:

```python
edges = [
    (0, 1, 3),
    (0, 2, 4),
    (1, 3, 5),
    (2, 3, 6)
]
```

---

### 🔄 Conversion from Adjacency Matrix to Adjacency List:

```python
import math

matrix = [
    [0, 3, 4, math.inf],
    [math.inf, 0, math.inf, 5],
    [math.inf, math.inf, 0, 6],
    [math.inf, math.inf, math.inf, 0]
]

n = len(matrix)
graph = {}

for i in range(n):
    graph[i] = {}
    for j in range(n):
        if matrix[i][j] != math.inf and matrix[i][j] != 0:
            graph[i][j] = matrix[i][j]

print(graph)
```

**Output:**

```python
{0: {1: 3, 2: 4}, 1: {3: 5}, 2: {3: 6}, 3: {}}
```

---

Would you like me to use this representation in the full max-flow Python code using Ford-Fulkerson?
