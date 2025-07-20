**Tarjan’s Algorithm** is used to find all **Strongly Connected Components (SCCs)** in a **directed graph** in **O(V + E)** time. A **strongly connected component** is a maximal group of vertices where each vertex is reachable from every other vertex in the group.

---

## 🧠 Intuition Behind Tarjan's Algorithm

Tarjan's algorithm uses:

* **DFS traversal**
* **Stack** to store the current path
* **Discovery time** and **low-link values**
* Identifies SCCs when a root node is found (where `disc[u] == low[u]`)

---

### 🔁 Terminology

* **Discovery time (`disc[u]`)**: When a node is first visited.
* **Low link value (`low[u]`)**: The lowest discovery time reachable from that node (including its subtree).

---

### ✅ Key Idea

If a node's discovery time equals its low-link value, then it is the **head of an SCC**, and all nodes above it in the recursion stack form an SCC.

---

## 🔧 Python Code: Tarjan’s Algorithm

```python
from collections import defaultdict

class TarjanSCC:
    def __init__(self, vertices):
        self.graph = defaultdict(list)
        self.V = vertices
        self.time = 0
        self.disc = [-1] * self.V
        self.low = [-1] * self.V
        self.stack_member = [False] * self.V
        self.stack = []
        self.SCCs = []

    def add_edge(self, u, v):
        self.graph[u].append(v)

    def _SCC_util(self, u):
        self.disc[u] = self.low[u] = self.time
        self.time += 1
        self.stack.append(u)
        self.stack_member[u] = True

        for v in self.graph[u]:
            if self.disc[v] == -1:
                self._SCC_util(v)
                self.low[u] = min(self.low[u], self.low[v])
            elif self.stack_member[v]:
                self.low[u] = min(self.low[u], self.disc[v])

        # If u is head of an SCC
        if self.low[u] == self.disc[u]:
            scc = []
            while True:
                w = self.stack.pop()
                scc.append(w)
                self.stack_member[w] = False
                if w == u:
                    break
            self.SCCs.append(scc)

    def find_SCCs(self):
        for i in range(self.V):
            if self.disc[i] == -1:
                self._SCC_util(i)
        return self.SCCs
```

---

## 🔍 Example Usage

```python
g = TarjanSCC(5)
g.add_edge(1, 0)
g.add_edge(0, 2)
g.add_edge(2, 1)
g.add_edge(0, 3)
g.add_edge(3, 4)

sccs = g.find_SCCs()
print("Strongly Connected Components:", sccs)
```

### 🔚 Output:

```
Strongly Connected Components: [[4], [3], [1, 2, 0]]
```

---

## 🧭 Time and Space Complexity

* **Time**: O(V + E)
* **Space**: O(V) for stack, discovery arrays, etc.

---

## 💡 Use Cases

* Analyzing cycles in dependency graphs
* Optimizing call graphs in compilers
* Finding isolated parts of software architecture

---

In graph theory, **cycles** and **connected components** refer to different structural properties of a graph. Here’s a clear breakdown:

---

## 🔁 **Cycle in a Graph**

### ✅ Definition:

A **cycle** is a path that starts and ends at the same vertex, with no repeated edges or vertices (except the start/end).

### 🧠 Key Points:

| Property    | Explanation                                        |
| ----------- | -------------------------------------------------- |
| Type        | Can occur in **directed** or **undirected** graphs |
| Requirement | Must have at least 3 nodes (in undirected)         |
| Check       | Use DFS or Union-Find to detect cycles             |
| Usage       | Detect deadlocks, feedback loops, etc.             |

### ➕ Example (Undirected):

```
1 -- 2
|    |
4 -- 3
```

This graph has a cycle: `1-2-3-4-1`

---

## 🔗 **Connected Components**

### ✅ Definition:

A **connected component** is a subgraph in which there is a path between **every pair of vertices**, and which is **disconnected** from the rest of the graph.

### 🧠 Key Points:

| Property   | Explanation                                                                       |
| ---------- | --------------------------------------------------------------------------------- |
| Type       | Applies to **undirected** or **strongly connected** components in directed graphs |
| Components | A graph may have multiple connected components                                    |
| Check      | Use DFS or BFS to find all nodes reachable from a node                            |
| Usage      | Cluster detection, network reliability, etc.                                      |

### ➕ Example:

```
Component 1:  1 -- 2 -- 3
Component 2:  4 -- 5
```

---

## 🔄 Difference Table

| Feature    | **Cycle**                                              | **Connected Component**                                              |
| ---------- | ------------------------------------------------------ | -------------------------------------------------------------------- |
| Definition | Path that loops back to starting node                  | Subgraph where all nodes are reachable from one another              |
| Graph Type | Directed or Undirected                                 | Typically undirected; directed has **strongly connected components** |
| Count      | A graph may have zero or many cycles                   | A graph has ≥1 connected components                                  |
| Detection  | Cycle detection algorithms (DFS back edge, Union-Find) | BFS/DFS starting from unvisited nodes                                |
| Use Case   | Detect loops/deadlocks                                 | Grouping connected regions, partitioning graph                       |

---

## 📌 Summary:

* A **cycle** shows **looping structure**.
* A **connected component** shows **reachability grouping**.
* A graph **can have connected components with or without cycles**.

---

Would you like code examples for detecting both in Python?

