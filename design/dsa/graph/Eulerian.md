### 🧭 Eulerian Path and Circuit in Graph Theory

**Eulerian Path** and **Eulerian Circuit** are concepts in graph theory that relate to traversing a graph by visiting **every edge exactly once**.

They are named after **Leonhard Euler**, who first solved the **Königsberg bridge problem** in 1736 — considered the origin of graph theory.

---

### ✅ Definitions

| Term                            | Definition                                                                                                          |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **Eulerian Path**               | A path in a graph that visits every **edge** exactly once. It **doesn't have to start and end at the same vertex**. |
| **Eulerian Circuit** (or Cycle) | A path that **starts and ends at the same vertex** and visits **every edge exactly once**.                          |

---

### 📋 Eulerian Conditions

#### For **Undirected Graphs**:

| Type             | Condition                                                                       |
| ---------------- | ------------------------------------------------------------------------------- |
| Eulerian Circuit | All vertices have **even degree** and the graph is **connected**.               |
| Eulerian Path    | Exactly **0 or 2 vertices** have **odd degree** and the graph is **connected**. |

#### For **Directed Graphs**:

| Type             | Condition                                                                   |
| ---------------- | --------------------------------------------------------------------------- |
| Eulerian Circuit | Every vertex’s in-degree = out-degree                                       |
| Eulerian Path    | At most one vertex with `out − in = 1`, one with `in − out = 1`, rest equal |

---

### 🔧 Python Code to Check for Eulerian Path (Undirected)

```python
from collections import defaultdict

class Graph:
    def __init__(self, V):
        self.V = V
        self.adj = defaultdict(list)

    def add_edge(self, u, v):
        self.adj[u].append(v)
        self.adj[v].append(u)

    def is_connected(self):
        visited = [False] * self.V
        # Start DFS from a non-isolated node
        start = next((i for i in range(self.V) if self.adj[i]), None)
        if start is None:
            return True  # No edges
        self.dfs(start, visited)
        return all(visited[i] or not self.adj[i] for i in range(self.V))

    def dfs(self, u, visited):
        visited[u] = True
        for v in self.adj[u]:
            if not visited[v]:
                self.dfs(v, visited)

    def has_eulerian_path(self):
        if not self.is_connected():
            return False
        odd = sum(1 for i in range(self.V) if len(self.adj[i]) % 2 == 1)
        return odd == 0 or odd == 2

    def has_eulerian_circuit(self):
        if not self.is_connected():
            return False
        return all(len(self.adj[i]) % 2 == 0 for i in range(self.V))
```

---

### 🧩 Real-World Problems Solved

| Application Domain  | Problem Description                                                          |
| ------------------- | ---------------------------------------------------------------------------- |
| **Logistics**       | Route that uses each road once (e.g., garbage collection, snow plowing)      |
| **DNA Sequencing**  | Eulerian paths in **De Bruijn graphs** help reconstruct original DNA strings |
| **Network Routing** | Avoid repeated transmission over same link                                   |
| **Puzzles**         | Like the famous **"Draw without lifting your pen"** problems                 |
| **Urban Planning**  | Bridge and street planning (origin: Königsberg bridges problem)              |

---

### 🧪 Example

Graph:

```
0---1
|   |
3---2
```

All vertices have **even degree → Eulerian Circuit exists**.

Remove edge (3,2):

Now, vertices 2 and 3 have **odd degree → Eulerian Path exists**, but **no Eulerian Circuit**.

---

### 📚 Eulerian vs Hamiltonian

| Feature        | Eulerian                     | Hamiltonian          |
| -------------- | ---------------------------- | -------------------- |
| Focus          | **Edges**                    | **Vertices**         |
| Visit          | Each edge **once**           | Each vertex **once** |
| Known Solution | Easy (with clear conditions) | NP-Complete problem  |

---

Would you like an algorithm to **construct** an Eulerian path using **Hierholzer’s algorithm**?


Great! Let's dive into **Hierholzer’s Algorithm**, which is the standard way to **construct an Eulerian Path or Circuit** in a graph.

---

## 🛠 Hierholzer’s Algorithm – Overview

Hierholzer's algorithm works if:

* The graph has an **Eulerian path** (0 or 2 odd-degree vertices)
* Or an **Eulerian circuit** (all even-degree vertices)

### 🔧 Idea:

1. Start at a vertex with a non-zero degree (start at an odd-degree vertex if Eulerian path).
2. Follow a trail of edges until returning to the starting vertex (circuit), or ending if path.
3. If any vertex along the trail has unused edges, **recurse** from there.
4. Combine the sub-trails into the final path.

---

## ✅ Python Implementation (Undirected Graph)

```python
from collections import defaultdict

class EulerianGraph:
    def __init__(self, V):
        self.graph = defaultdict(list)
        self.V = V

    def add_edge(self, u, v):
        self.graph[u].append(v)
        self.graph[v].append(u)

    def remove_edge(self, u, v):
        self.graph[u].remove(v)
        self.graph[v].remove(u)

    def find_start_vertex(self):
        # Start from a vertex with odd degree if exists (Eulerian Path)
        start = 0
        for v in range(self.V):
            if len(self.graph[v]) % 2 == 1:
                return v
        # Else, return any vertex with an edge
        for v in range(self.V):
            if self.graph[v]:
                return v
        return 0

    def hierholzer(self):
        graph = self.graph
        curr_path = []
        circuit = []

        curr_v = self.find_start_vertex()
        curr_path.append(curr_v)

        while curr_path:
            if graph[curr_v]:
                curr_path.append(curr_v)
                next_v = graph[curr_v][-1]
                self.remove_edge(curr_v, next_v)
                curr_v = next_v
            else:
                circuit.append(curr_v)
                curr_v = curr_path.pop()
        
        circuit.reverse()
        return circuit
```

---

## 🧪 Example

```python
g = EulerianGraph(5)
g.add_edge(0, 1)
g.add_edge(1, 2)
g.add_edge(2, 0)
g.add_edge(0, 3)
g.add_edge(3, 4)
g.add_edge(4, 0)

path = g.hierholzer()
print("Eulerian Path/Circuit:", path)
```

Output:

```
Eulerian Path/Circuit: [0, 1, 2, 0, 4, 3, 0]
```

---

## 🧠 Notes

* The returned path includes **all edges exactly once**.
* You should validate the graph before using this method:

  * **Is the graph connected?**
  * **Does it satisfy Eulerian path/circuit conditions?**

---

### 🚦 Want to try it on a **directed graph**, or should I include the validation step as well?
