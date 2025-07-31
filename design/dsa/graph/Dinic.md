**Dinic's Algorithm** (also called **Dinitz's Algorithm**) is an efficient algorithm used to solve the **Maximum Flow problem** in a **flow network**. It improves upon the Ford-Fulkerson method using **level graphs** and **blocking flows** to avoid redundant searches.

---

## 🧠 Key Concepts

1. **Flow Network**: A directed graph where each edge has a capacity, and we send flow from a source `s` to a sink `t`.

2. **Level Graph**: Built using **BFS**, it assigns a level to each node which is the **shortest distance (in edges)** from the source.

3. **Blocking Flow**: A flow that **cannot be increased further** in the current level graph. Built using **DFS**.

4. **Dinic's Algorithm Steps**:

   * Repeat:

     1. Build the level graph using BFS from `s`.
     2. While there is a blocking flow, push flow using DFS.
   * Until there is **no path from `s` to `t`**.

---

## 🕒 Time Complexity

* **O(V²E)** in general
* **O(E√V)** for bipartite graphs
* **Much faster in practice** than naive Ford-Fulkerson

---

## ✅ Python Code for Dinic's Algorithm

```python
from collections import deque

class Edge:
    def __init__(self, to, rev, capacity):
        self.to = to
        self.rev = rev
        self.cap = capacity

class Dinic:
    def __init__(self, n):
        self.n = n
        self.graph = [[] for _ in range(n)]

    def add_edge(self, fr, to, cap):
        forward = Edge(to, len(self.graph[to]), cap)
        backward = Edge(fr, len(self.graph[fr]), 0)
        self.graph[fr].append(forward)
        self.graph[to].append(backward)

    def bfs_level(self, s, t, level):
        queue = deque()
        level[s] = 0
        queue.append(s)
        while queue:
            v = queue.popleft()
            for e in self.graph[v]:
                if e.cap > 0 and level[e.to] < 0:
                    level[e.to] = level[v] + 1
                    queue.append(e.to)
        return level[t] != -1

    def dfs_flow(self, v, t, upTo, iter, level):
        if v == t:
            return upTo
        for i in range(iter[v], len(self.graph[v])):
            e = self.graph[v][i]
            if e.cap > 0 and level[v] < level[e.to]:
                d = self.dfs_flow(e.to, t, min(upTo, e.cap), iter, level)
                if d > 0:
                    e.cap -= d
                    self.graph[e.to][e.rev].cap += d
                    return d
            iter[v] += 1
        return 0

    def max_flow(self, s, t):
        flow = 0
        INF = float('inf')
        level = [-1] * self.n
        while self.bfs_level(s, t, level):
            iter = [0] * self.n
            f = self.dfs_flow(s, t, INF, iter, level)
            while f > 0:
                flow += f
                f = self.dfs_flow(s, t, INF, iter, level)
            level = [-1] * self.n
        return flow
```

---

## 🧪 Example Usage

```python
# Create Dinic object with 4 nodes (0 to 3)
d = Dinic(4)

# Add edges: from, to, capacity
d.add_edge(0, 1, 3)
d.add_edge(0, 2, 2)
d.add_edge(1, 2, 1)
d.add_edge(1, 3, 2)
d.add_edge(2, 3, 4)

# Compute max flow from source (0) to sink (3)
print("Max Flow:", d.max_flow(0, 3))  # Output should be 5
```

---

## 🔁 Flow Update Logic

* When an edge sends flow, its capacity decreases
* The **reverse edge** gets increased capacity (residual capacity)

---

## 📌 Applications

* Network routing
* Bipartite matching
* Project selection problems
* Image segmentation (in computer vision)

---

Would you like a **visual step-by-step explanation** of how the algorithm runs on a graph?
