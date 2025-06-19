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
