**Kahn’s Algorithm** is used to perform **Topological Sorting** of a **Directed Acyclic Graph (DAG)**. It can also be used to detect cycles in a directed graph.

---

### ✅ Key Concepts

* **Topological Sort**: Linear ordering of vertices such that for every directed edge `u → v`, vertex `u` comes before `v` in the ordering.
* Kahn’s Algorithm is **BFS-based** and uses **in-degree** (number of incoming edges) of each vertex.
* If at the end, topological sort includes **fewer nodes than total nodes**, a **cycle** exists.

---

### 🔧 Steps

1. Calculate **in-degree** of all nodes.
2. Add all nodes with **in-degree 0** to a queue.
3. While queue is not empty:

   * Pop node from queue, add to result.
   * For each neighbor of node:

     * Reduce its in-degree by 1.
     * If in-degree becomes 0, add to queue.
4. If result size != number of nodes → graph has a cycle.

---

### 🧪 Python Code

```python
from collections import deque, defaultdict

def kahn_topological_sort(graph):
    in_degree = defaultdict(int)
    topo_order = []

    # Initialize in-degree of all nodes
    for u in graph:
        for v in graph[u]:
            in_degree[v] += 1

    # Queue for nodes with in-degree 0
    queue = deque([u for u in graph if in_degree[u] == 0])

    while queue:
        node = queue.popleft()
        topo_order.append(node)

        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(topo_order) != len(graph):
        print("Cycle detected! No topological sort possible.")
        return None

    return topo_order
```

---

### 🔍 Example:

```python
graph = {
    'A': ['C'],
    'B': ['C', 'D'],
    'C': ['E'],
    'D': ['F'],
    'E': ['H', 'F'],
    'F': ['G'],
    'G': [],
    'H': []
}

result = kahn_topological_sort(graph)
print("Topological Sort:", result)
```

---

### 📌 Output:

```
Topological Sort: ['A', 'B', 'C', 'D', 'E', 'H', 'F', 'G']
```

---

### 🧠 Applications:

* Build systems (resolving dependencies)
* Course scheduling (prerequisites)
* Job/task scheduling
* Detecting cycles in a directed graph (no topological order if cyclic)

Would you like a version that handles disconnected graphs or returns all valid topological sorts?
