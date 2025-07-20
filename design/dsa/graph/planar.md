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


 **real-world applications** of both **planar** and **non-planar graphs**, and **why planarity matters** in system design, circuit layouts, geography, and more:

---

### ✅ **Applications of Planar Graphs**

Planar graphs can be drawn on a 2D plane without any edges crossing — this property is useful in systems that need clarity, low complexity, or physical constraints (like wiring or drawing).

#### 🔹 1. **VLSI Design (Chip Layout)**

* **Why planar?** Crossings increase complexity, cost, and error risk.
* Planar graphs are used to route wires on a chip without overlaps.
* Planarity testing is important for designing single-layer PCB (printed circuit boards).

#### 🔹 2. **Geographic Information Systems (GIS)**

* Used to model road networks, city layouts, rivers, pipelines, etc.
* Roads and streets can often be approximated with planar graphs.
* Algorithms on planar graphs run faster (e.g., shortest path on planar road maps).

#### 🔹 3. **Graph Drawing and Network Visualization**

* Planar graphs help in making clean, readable drawings of networks (e.g., org charts, metro maps).
* Tools like **graphviz** try to minimize edge crossings even for non-planar graphs.

#### 🔹 4. **Mesh Generation (in 3D Modeling and FEM Simulations)**

* Planar graphs are used to create **2D meshes** for simulations and finite element modeling.

#### 🔹 5. **Map Coloring Problem**

* The **Four Color Theorem** says any planar map can be colored with 4 colors — used in cartography and conflict-free resource assignment.

---

### 🚫 **Applications of Non-Planar Graphs**

Non-planar graphs arise naturally in more complex or high-connectivity systems where edge crossings are inevitable.

#### 🔸 1. **Social Networks**

* A user might be connected to thousands of others.
* These graphs are highly dense and rarely planar.

#### 🔸 2. **Computer Networks (like the Internet)**

* Routers and data centers form large, dense connection graphs.
* Non-planar due to multiple redundant paths, cross-links for reliability.

#### 🔸 3. **Transportation Networks (Airlines, Railways)**

* Unlike roads, airlines fly over each other and cross continents.
* Airline route networks are modeled as **non-planar** due to high connectivity.

#### 🔸 4. **Biological Networks**

* Protein interaction networks, gene regulatory networks — usually non-planar and very dense.
* Require specialized visualization techniques.

#### 🔸 5. **Electric Grid / Power Distribution**

* High-voltage transmission networks form dense meshes to ensure redundancy and reliability.
* Graph is non-planar due to many crossing paths.

#### 🔸 6. **3D Circuit Design / Multi-layer PCB**

* When layout cannot be embedded in 2D without overlaps, it becomes non-planar — handled using multi-layer PCBs.

---

### 🧠 Why Does Planarity Matter?

| Factor            | Planar Graphs                   | Non-Planar Graphs                     |
| ----------------- | ------------------------------- | ------------------------------------- |
| **Visualization** | Easy to draw and interpret      | Difficult due to edge crossings       |
| **Algorithms**    | Specialized, faster algorithms  | General algorithms, higher complexity |
| **Layout design** | Cheaper and simpler wiring      | Requires more layers or rerouting     |
| **Use case**      | Road maps, circuits, city plans | Internet, social networks, airlines   |

---

### 🛠️ Want to Try It Out?

You can use Python’s [`networkx`](https://networkx.org/) to:

* Check if a graph is planar
* Generate planar and non-planar graphs
* Visualize them using `matplotlib`

Let me know if you want that sample code.

---

Would you like to go deeper into **planarity testing algorithms** or **Kuratoswki’s Theorem**?
