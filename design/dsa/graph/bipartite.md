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


### 🎯 What is **Bipartite Matching** in Graph Theory?

**Bipartite Matching** is a concept used in **bipartite graphs**, which are graphs whose **vertices can be divided into two disjoint sets** `U` and `V` such that every edge connects a vertex in `U` to one in `V`.

---

### ✅ **Definition**

A **matching** is a set of edges such that no two edges share a common vertex.

A **bipartite matching** is a matching where:

* The graph is **bipartite**.
* Each edge connects a vertex from one partition to another.
* The goal is usually to **maximize the number of matched pairs**.

---

### ✅ **Real-world Examples**

1. **Job Assignment**: Workers on one side, jobs on the other; edges indicate who can do what.
2. **Student–Project Allocation**: Students and projects as sets, with edges showing interest or eligibility.
3. **Stable Marriage Problem**: Men and women matched based on preferences.

---

### ✅ **Types of Bipartite Matching**

| Type                    | Description                                              |
| ----------------------- | -------------------------------------------------------- |
| **Maximum Matching**    | The largest possible set of matched edges.               |
| **Perfect Matching**    | Every vertex in both partitions is matched.              |
| **Maximum Cardinality** | Matching with the most edges (same as Maximum Matching). |

---

### ✅ **Common Algorithms**

| Algorithm                        | Description                                               | Time Complexity           |
| -------------------------------- | --------------------------------------------------------- | ------------------------- |
| **Hungarian Algorithm**          | Used for weighted bipartite matching (assignment problem) | `O(n³)`                   |
| **Hopcroft–Karp Algorithm**      | Efficient for unweighted bipartite matching               | `O(√V * E)`               |
| **Ford–Fulkerson (as Max Flow)** | Can be used to solve bipartite matching via max flow      | Depends on implementation |

---

### 🧠 Graph Example

Set `U = {u1, u2, u3}`, Set `V = {v1, v2, v3}`
Edges = `{(u1,v2), (u2,v1), (u3,v1), (u3,v3)}`

A possible **maximum matching** is:

* `(u1,v2)`
* `(u2,v1)`
* `(u3,v3)`

---

### 🧪 Python Code: Using NetworkX

```python
import networkx as nx

# Create a bipartite graph
B = nx.Graph()

# Add nodes with bipartite attribute
U = {"u1", "u2", "u3"}
V = {"v1", "v2", "v3"}
B.add_nodes_from(U, bipartite=0)
B.add_nodes_from(V, bipartite=1)

# Add edges between U and V
edges = [("u1", "v2"), ("u2", "v1"), ("u3", "v1"), ("u3", "v3")]
B.add_edges_from(edges)

# Find maximum matching
matching = nx.bipartite.maximum_matching(B, top_nodes=U)
print("Matching:", matching)
```

---

Would you like a visualization or real-world scenario implemented with this?
