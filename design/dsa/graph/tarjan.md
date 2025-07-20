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

Would you like a visualization or example on a real-world use case like package dependency resolution?
