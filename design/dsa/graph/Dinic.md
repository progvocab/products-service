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
---
A **level graph** is a concept that comes up mainly in **graph theory and network flow algorithms** (especially **Dinic’s algorithm**).

---

## 🔹 Definition

A **level graph** is a graph derived from a flow network by assigning **levels (distances)** to vertices from a chosen **source**:

1. Perform a **BFS (Breadth-First Search)** from the source node `s`.
2. Each vertex gets a **level = shortest distance (in edges) from `s`**.
3. Keep only edges that go **from a vertex at level `i` to a vertex at level `i+1`**.
   (Edges that don’t respect this direction are removed.)

The result is the **level graph**.

---

## 🔹 Why is it useful?

In **Dinic’s algorithm** for **Maximum Flow**:

* The level graph ensures we only push flow along **shortest augmenting paths**.
* This speeds up the algorithm, avoiding wasted searches.

---

## 🔹 Example

Suppose we have a flow network:

```
Source (S) → A → C → Sink (T)
       ↓    ↓
        B → D
```

### Step 1: BFS from `S`

* Level(S) = 0
* Level(A) = 1, Level(B) = 1
* Level(C) = 2, Level(D) = 2
* Level(T) = 3

### Step 2: Build Level Graph

* Keep only edges going from level `i` → level `i+1`:

  * S → A, S → B
  * A → C, B → D
  * C → T, D → T
* Remove edges like `A → D` if they skip levels.

Now the level graph looks like layers:

```
Level 0:   S
Level 1:   A   B
Level 2:   C   D
Level 3:   T
```

---

## 🔹 Summary

* **Level graph** = layered version of original graph (BFS from source).
* Used in **Dinic’s maximum flow algorithm**.
* Ensures augmenting paths are shortest, improving efficiency.

---

⚡ Would you like me to also **draw the difference** between the **original graph** and the **level graph** for a small flow example, so it’s more visual?
