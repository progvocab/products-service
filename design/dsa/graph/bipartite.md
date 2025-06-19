A **bipartite graph** (or **bigraph**) is a special kind of graph where:

> **The set of vertices can be divided into two disjoint sets such that no two vertices within the same set are adjacent.**

---

## 🔹 Formal Definition

A graph **G(V, E)** is **bipartite** if its vertex set **V** can be partitioned into **two sets** `U` and `V` such that:

* Every edge connects a vertex in `U` to one in `V`.
* No edge exists between vertices **within** the same set.

---

## 🎯 Real-World Examples

| Scenario                                  | Bipartite Sets     |
| ----------------------------------------- | ------------------ |
| Job assignments                           | Workers and Jobs   |
| Course prerequisites                      | Courses and Topics |
| Social network (people-likes-pages)       | People and Pages   |
| Matching problems (e.g., stable marriage) | Men and Women      |

---

## 🔸 Visual Example

```
  Set U       Set V
   A   B       1   2
    \ /         \ /
     C           3

Edges: A-1, A-2, B-2, C-3
```

* Vertices: {A, B, C, 1, 2, 3}
* Partition: U = {A, B, C}, V = {1, 2, 3}
* No edges within U or V → ✅ Bipartite

---

## 🔸 How to Check if a Graph is Bipartite?

You can use **BFS or DFS** to color the graph with **2 colors**:

* If adjacent nodes **always get different colors**, the graph is bipartite.
* If at any point two adjacent nodes have the **same color**, it’s **not** bipartite.

---

## ✅ Python Example Using BFS

```python
from collections import deque

def is_bipartite(graph):
    color = {}
    for node in graph:
        if node not in color:
            queue = deque([node])
            color[node] = 0
            while queue:
                u = queue.popleft()
                for v in graph[u]:
                    if v not in color:
                        color[v] = 1 - color[u]
                        queue.append(v)
                    elif color[v] == color[u]:
                        return False
    return True

# Example graph (adjacency list)
graph = {
    0: [1, 3],
    1: [0, 2],
    2: [1, 3],
    3: [0, 2]
}

print(is_bipartite(graph))  # True
```

---

## 🔄 Bipartite vs Non-Bipartite

| Graph                        | Bipartite? |
| ---------------------------- | ---------- |
| A triangle (3-cycle)         | ❌          |
| Even cycle (e.g., square)    | ✅          |
| Star (1 center, many leaves) | ✅          |

---

Let me know if you’d like to:

* See **maximum matching** in bipartite graphs
* Use the **Hopcroft-Karp** algorithm
* Convert real-world problems into bipartite models
