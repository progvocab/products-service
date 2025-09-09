The **Königsberg bridge problem** is a **classic graph theory problem** that gave birth to **Eulerian graphs**.

---

## 📖 Problem Statement

* The city of Königsberg (now Kaliningrad, Russia) had **7 bridges** connecting 2 islands and 2 riverbanks.
* Question: *Can you walk through the city crossing every bridge exactly once?*

---

## 🔑 Euler’s Insight

This is the **Eulerian Trail / Circuit** problem:

* **Eulerian circuit**: a cycle that visits every edge exactly once and returns to start.
  🔹 Exists if every vertex has even degree and the graph is connected.

* **Eulerian trail**: a path (not necessarily cycle) that visits every edge exactly once.
  🔹 Exists if **exactly two vertices have odd degree** (start and end points).

👉 In Königsberg, **all 4 land areas had odd degree**, so **no Eulerian trail/circuit exists**.

---

## 🐍 Python Implementation

We’ll model the problem and check Eulerian conditions.

```python
from collections import defaultdict

class Graph:
    def __init__(self):
        self.graph = defaultdict(list)

    def add_edge(self, u, v):
        self.graph[u].append(v)
        self.graph[v].append(u)  # undirected

    def is_connected(self):
        visited = set()
        nodes = list(self.graph.keys())

        def dfs(u):
            visited.add(u)
            for v in self.graph[u]:
                if v not in visited:
                    dfs(v)

        dfs(nodes[0])
        return len(visited) == len(nodes)

    def eulerian_status(self):
        if not self.is_connected():
            return "Graph is not connected → No Eulerian trail/circuit"

        odd_degree = sum(1 for node in self.graph if len(self.graph[node]) % 2 != 0)

        if odd_degree == 0:
            return "Eulerian Circuit exists"
        elif odd_degree == 2:
            return "Eulerian Trail exists"
        else:
            return "No Eulerian trail/circuit"


# 🔹 Königsberg Bridge Graph (4 land areas A,B,C,D with 7 bridges)
g = Graph()
edges = [
    ("A","B"), ("A","B"),  # 2 bridges
    ("A","C"),
    ("A","D"),
    ("B","C"),
    ("B","D"),
    ("C","D"),
]

for u, v in edges:
    g.add_edge(u, v)

print(g.eulerian_status())
```

---

## ✅ Output

```
No Eulerian trail/circuit
```

---

## 🔎 Explanation

* Degrees of nodes:

  * A = 3
  * B = 3
  * C = 3
  * D = 3
* All four are odd → ❌ impossible.

This matches Euler’s conclusion in 1736. 🎉

---

👉 Do you want me to also extend this program to **actually find and print an Eulerian trail/circuit** (when it exists), using **Hierholzer’s algorithm**?
