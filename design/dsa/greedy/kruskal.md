### 🌳 Kruskal's Algorithm — Minimum Spanning Tree (MST)

**Kruskal’s Algorithm** is a greedy algorithm used to find the **Minimum Spanning Tree (MST)** of a **connected, undirected, weighted graph**.

---

### 🧠 Goal

Given a graph `G(V, E)`, the goal is to find a **subset of edges** that:

* Connects all vertices (spanning tree)
* Has **minimum possible total edge weight**
* Contains **no cycles**

---

### 🔧 Core Idea

1. **Sort** all edges by **increasing weight**.
2. Initialize a **disjoint set (Union-Find)** to track connected components.
3. Iterate over edges:

   * If the edge connects two **different components** → **include** it in MST.
   * Otherwise → **skip** it (to avoid cycle).
4. Stop when you have **V - 1 edges** in the MST.

---

### ✅ Kruskal’s Algorithm Steps

```text
Input: Graph G(V, E)
Output: Minimum Spanning Tree (set of edges)
1. Sort all edges in increasing order of weight.
2. Initialize disjoint sets for each vertex.
3. For each edge (u, v) in sorted order:
   a. If find(u) != find(v):  // no cycle
       - Add edge (u, v) to MST
       - union(u, v)
4. Repeat until MST has (V - 1) edges
```

---

### 🧮 Python Code Example

```python
class DisjointSet:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return False
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
        return True

def kruskal(n, edges):
    # edges: List of (weight, u, v)
    edges.sort()
    ds = DisjointSet(n)
    mst = []
    total_weight = 0

    for weight, u, v in edges:
        if ds.union(u, v):
            mst.append((u, v, weight))
            total_weight += weight
        if len(mst) == n - 1:
            break

    return mst, total_weight
```

---

### 📌 Example

Graph:

```
  1
A---B
|\  |
| \ |
4  \| 3
|   C
2   \
|    \
D-----E
   5
```

Edges:

```
(A, B, 1)
(A, D, 2)
(A, C, 4)
(B, C, 3)
(D, E, 5)
```

Converted to 0-based:

```python
edges = [
    (1, 0, 1),  # A-B
    (2, 0, 3),  # A-D
    (4, 0, 2),  # A-C
    (3, 1, 2),  # B-C
    (5, 3, 4),  # D-E
]
mst, cost = kruskal(5, edges)
print("MST:", mst)
print("Total Cost:", cost)
```

Output:

```
MST: [(0, 1, 1), (0, 3, 2), (1, 2, 3), (3, 4, 5)]
Total Cost: 11
```

---

### 🧠 Time Complexity

| Step           | Complexity                             |
| -------------- | -------------------------------------- |
| Sorting edges  | O(E log E)                             |
| Union-Find ops | O(α(N)) \~ constant with optimizations |
| Total          | **O(E log E)** — dominant step         |

---

### ✅ Use Cases

* **Network Design** (cables, pipelines, roads)
* **Clustering Algorithms**
* **Image Segmentation**

---

Would you like a **visual diagram** or a comparison with **Prim’s Algorithm** too?
