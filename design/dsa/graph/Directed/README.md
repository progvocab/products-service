 **graph theory basics** about **directed trails, paths, circuits, and cycles**.

---

# 🔹 Preliminaries

We are talking about a **directed graph (digraph)** $G = (V,E)$ where:

* $V$ = set of vertices (nodes)
* $E$ = set of directed edges (arcs), where each edge has a direction $u \to v$.

---

# 🔹 1. Directed Walk

A **walk** is just a sequence of vertices connected by edges.

Example:
In graph $A \to B \to C \to A$,
the walk could be $A \to B \to C \to A \to B$.

👉 No restriction: vertices and edges may repeat.

---

# 🔹 2. Directed Trail

A **trail** is a walk where **edges are not repeated**, but vertices may repeat.

Example:
$A \to B \to C \to A$ is a trail (each edge used once).
But $A \to B \to C \to A \to B$ is **not a trail**, since edge $A \to B$ is reused.

---

# 🔹 3. Directed Path

A **path** is a trail where **no vertex is repeated** (except possibly start=end in a cycle).

Example:
$A \to B \to C$ is a path.
$A \to B \to C \to A$ is **not** a path (since vertices repeat).

👉 Paths are “simple, non-repeating” routes.

---

# 🔹 4. Directed Circuit

A **circuit** (closed trail) is a trail that starts and ends at the same vertex.

* Edges are not repeated
* Vertices may repeat (except start=end)

Example:
$A \to B \to C \to A$ is a circuit.

---

# 🔹 5. Directed Cycle

A **cycle** is a circuit where **no vertex is repeated** except the starting/ending one.

Example:
$A \to B \to C \to A$ is a cycle.
$A \to B \to C \to A \to D \to A$ is a circuit but not a cycle (since `A` appears more than once inside).

---

# 🔹 Summary Table

| Term    | Start=End? | Repeats Vertices?                | Repeats Edges? |
| ------- | ---------- | -------------------------------- | -------------- |
| Walk    | Maybe      | Allowed                          | Allowed        |
| Trail   | Maybe      | Allowed                          | ❌ No           |
| Path    | Maybe      | ❌ No (except start=end in cycle) | ❌ No           |
| Circuit | ✅ Yes      | Allowed                          | ❌ No           |
| Cycle   | ✅ Yes      | ❌ No (except start=end)          | ❌ No           |

---

# 🔹 Python Representation Example

```python
import networkx as nx

# Directed graph
G = nx.DiGraph()
edges = [("A","B"), ("B","C"), ("C","A"), ("C","D")]
G.add_edges_from(edges)

print("Nodes:", G.nodes())
print("Edges:", G.edges())

# Check if a path exists
path = nx.has_path(G, "A", "D")
print("Path A->D exists?", path)

# Find simple paths
all_paths = list(nx.all_simple_paths(G, "A", "D"))
print("All simple paths A->D:", all_paths)

# Find cycles
cycles = list(nx.simple_cycles(G))
print("Cycles:", cycles)
```

---

# 🔹 Example Output

```
Nodes: ['A', 'B', 'C', 'D']
Edges: [('A', 'B'), ('B', 'C'), ('C', 'A'), ('C', 'D')]
Path A->D exists? True
All simple paths A->D: [['A', 'B', 'C', 'D']]
Cycles: [['A', 'B', 'C']]
```

---

✅ So:

* **Trail** = no edge repeats
* **Path** = no vertex repeats
* **Circuit** = closed trail
* **Cycle** = closed path

---

Would you like me to also explain this visually with a **diagram of a directed graph** showing a trail, path, circuit, and cycle?
