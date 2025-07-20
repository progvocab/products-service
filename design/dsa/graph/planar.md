To determine whether a graph is **planar** in Python, you can use the **`networkx`** library, which provides a built-in function to check for **planarity** using the **Kuratowski’s theorem**-based algorithm.

---

## ✅ Step-by-Step: Check if a Graph is Planar in Python

### 1. 📦 Install NetworkX

```bash
pip install networkx
```

---

### 2. ✅ Use `check_planarity()` function

```python
import networkx as nx

# Create a graph
G = nx.Graph()

# Add edges
G.add_edges_from([
    (1, 2), (2, 3), (3, 4), (4, 5),
    (5, 1), (1, 3), (2, 4)  # This adds a K5 subgraph (non-planar)
])

# Check if the graph is planar
is_planar, _ = nx.check_planarity(G)

print("Is the graph planar?", is_planar)
```

---

## 📌 Output

```
Is the graph planar? False
```

---

## 🔍 Explanation

* `check_planarity(G)` returns a tuple:

  * `True`/`False` — if graph is planar.
  * A `PlanarEmbedding` object if it is planar.

---

## 🧠 Theoretical Note

A graph is **non-planar** if it contains:

* A subgraph homeomorphic to **K₅** (complete graph on 5 nodes)
* Or to **K₃,₃** (complete bipartite graph on 3 + 3 nodes)

These are **Kuratowski’s Theorem** obstructions.

---

## 🧪 Example: Planar Graph

```python
G = nx.cycle_graph(5)  # A simple cycle
print(nx.check_planarity(G)[0])  # True
```

---

## 📚 Optional: Visualize the Graph

```python
import matplotlib.pyplot as plt

nx.draw(G, with_labels=True)
plt.show()
```

---

Would you like to extract the planar embedding or visualize the forbidden subgraph (K5/K3,3) when it’s not planar?

In graph theory, graphs denoted using the letter **K** are **complete graphs** or **complete bipartite graphs**, which are key in understanding **planarity, coloring, and graph structure**.

Let’s break it down:

---

## ✅ 1. **Complete Graphs (K<sub>n</sub>)**

### 🔹 Definition:

A **complete graph** on `n` vertices is a simple undirected graph where **every pair of distinct vertices is connected by a unique edge**.

### 📘 Notation:

* **K<sub>n</sub>**: Complete graph with `n` vertices.
* Total number of edges = `n(n - 1) / 2`

### 📊 Examples:

| Name   | Vertices | Edges | Planar?           |
| ------ | -------- | ----- | ----------------- |
| K₁     | 1        | 0     | Yes               |
| K₂     | 2        | 1     | Yes               |
| K₃     | 3        | 3     | Yes               |
| K₄     | 4        | 6     | Yes               |
| **K₅** | 5        | 10    | ❌ No (Not planar) |

---

### 🔺 K<sub>5</sub>: Complete Graph with 5 Nodes

```
Each node connects to all others.
```

```
     1
   / | \
  2--3--4
   \ | /
     5
```

* Total Edges: 10
* **K₅ is the smallest complete graph that is not planar** (can’t be drawn on a plane without edge crossings).

---

## ✅ 2. **Complete Bipartite Graphs (K<sub>m,n</sub>)**

### 🔹 Definition:

A **complete bipartite graph** divides nodes into two disjoint sets with `m` and `n` nodes such that:

* Every node in set 1 is connected to **every node** in set 2.
* No edges **within** a set.

### 📘 Notation:

* **K<sub>m,n</sub>**: Bipartite graph with parts of size `m` and `n`
* Total edges = `m × n`

### 📊 Examples:

| Name                | m | n | Edges | Planar?           |
| ------------------- | - | - | ----- | ----------------- |
| K<sub>1,1</sub>     | 1 | 1 | 1     | Yes               |
| K<sub>2,2</sub>     | 2 | 2 | 4     | Yes               |
| **K<sub>3,3</sub>** | 3 | 3 | 9     | ❌ No (Not planar) |

---

### 🔺 K<sub>3,3</sub>: Bipartite with Two Sets of 3 Nodes

```
Set A: {A1, A2, A3}
Set B: {B1, B2, B3}
Every Ai connects to every Bi.
```

```
A1 —— B1
 | \   | \
A2 —— B2
 |     |
A3 —— B3
```

* **K₃,₃ is not planar** (classic utility problem: connect 3 houses to 3 utilities without crossing wires — impossible on a plane)

---

## 🧠 Summary Table

| Graph Type         | Notation        | Description                                          | Planar Limit                          |
| ------------------ | --------------- | ---------------------------------------------------- | ------------------------------------- |
| Complete Graph     | Kₙ              | Every node connected to all others                   | Planar only for **n ≤ 4**             |
| Complete Bipartite | K<sub>m,n</sub> | Each node in one set connects to every node in other | Planar only if **m ≤ 2** or **n ≤ 2** |

---

## 🔎 Applications

* **K₅** and **K₃,₃** are **obstructions to planarity** (Kuratowski’s Theorem)
* Used in **network design**, **social networks**, **graph coloring**, etc.

---

Let me know if you’d like visualizations or Python code to generate K<sub>n</sub> and K<sub>m,n</sub> graphs using `networkx`.
