

# 🌐 Floyd–Warshall Algorithm

The **Floyd–Warshall algorithm** is a **graph algorithm** used to find the **shortest paths between all pairs of vertices** in a **weighted graph** (directed or undirected).

It can handle **negative edge weights** (but not negative cycles).

---

## 1. **Problem Statement**

Given a graph $G(V, E)$ with:

* $V$: set of vertices
* $E$: set of weighted edges

We want to compute the matrix `dist[i][j]`, where each entry is the **shortest distance from vertex $i$ to vertex $j$**.

---

## 2. **Core Idea**

The algorithm is based on **dynamic programming**.

Let $dist[i][j]$ = shortest distance from $i$ to $j$.

For each vertex $k$, we check:

* Is the path $i \to k \to j$ shorter than the current $i \to j$?

If yes, we update:

$$
dist[i][j] = \min(dist[i][j], \; dist[i][k] + dist[k][j])
$$

We repeat this for all vertices $k$ (intermediate nodes).

---

## 3. **Algorithm Steps**

1. Initialize `dist[i][j]` with edge weights:

   * If $i = j$, `dist[i][j] = 0`
   * If edge $i \to j$ exists, `dist[i][j] = weight(i,j)`
   * Else, `dist[i][j] = ∞`

2. For each vertex $k = 1$ to $n$:
   For each pair $(i, j)$:
   Update:

   $$
   dist[i][j] = \min(dist[i][j], dist[i][k] + dist[k][j])
   $$

---

## 4. **Example**

Graph with 4 vertices:

| From → To | 1 | 2 | 3 | 4  |
| --------- | - | - | - | -- |
| **1**     | 0 | 5 | ∞ | 10 |
| **2**     | ∞ | 0 | 3 | ∞  |
| **3**     | ∞ | ∞ | 0 | 1  |
| **4**     | ∞ | ∞ | ∞ | 0  |

* Start with this as `dist`.
* After running Floyd–Warshall, we get the shortest paths between **all pairs**.

For example,

* Shortest path from 1 → 3 becomes **8** (via 2).
* Shortest path from 1 → 4 becomes **9** (via 2 → 3).

---

## 5. **Pseudocode**

```python
def floyd_warshall(graph):
    n = len(graph)
    dist = [[graph[i][j] for j in range(n)] for i in range(n)]

    for k in range(n):  # intermediate vertex
        for i in range(n):  # source
            for j in range(n):  # destination
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    return dist
```

---

## 6. **Complexity**

* **Time Complexity**: $O(V^3)$ (since we have 3 nested loops).
* **Space Complexity**: $O(V^2)$ (for the distance matrix).

---

## 7. **Applications**

* Finding **all-pairs shortest paths** in dense graphs.
* **Routing algorithms** in networks.
* **Social network analysis** (shortest connection between people).
* **Game theory / AI** (evaluating possible moves with multiple paths).

---

✅ **Summary**: The Floyd–Warshall algorithm is a dynamic programming approach to compute shortest paths between every pair of nodes in a graph. It’s elegant, simple, and works well for graphs with up to a few hundred vertices.

---

Would you like me to **show you a worked-out step-by-step example with matrices** (updating after each $k$) so you see how the paths improve iteratively?


Good follow-up 👍

The **Floyd–Warshall algorithm** works for **both directed and undirected graphs**, but with a small difference in how you **initialize** the adjacency (distance) matrix.

---

### 🔹 1. Directed Graph

* Each edge has a **direction**.
* If edge is $u \to v$ with weight $w$, then:

  $$
  dist[u][v] = w
  $$
* But **not** necessarily $dist[v][u]$.

Example:
Edge $1 \to 2$ with weight 3:

* `dist[1][2] = 3`
* `dist[2][1] = ∞` (unless explicitly given).

---

### 🔹 2. Undirected Graph

* Each edge has **no direction** (symmetric).
* If edge is $(u, v)$ with weight $w$, then:

  $$
  dist[u][v] = dist[v][u] = w
  $$

Example:
Edge $(1,2)$ with weight 3:

* `dist[1][2] = 3`
* `dist[2][1] = 3`

---

### 🔹 3. The Core Algorithm

The **triple loop update**:

$$
dist[i][j] = \min(dist[i][j],\; dist[i][k] + dist[k][j])
$$

is **identical** for both directed and undirected graphs ✅.

The **only difference** is how you set the initial `dist` matrix.

---

### 🔹 4. Example

* **Directed graph**: `1 → 2` (weight 5), no reverse edge

  * `dist[1][2] = 5`, `dist[2][1] = ∞`

* **Undirected graph**: `1 — 2` (weight 5)

  * `dist[1][2] = 5`, `dist[2][1] = 5`

Then the algorithm runs the same way.

---

✅ **Conclusion**:
Yes, Floyd–Warshall works for both **directed** and **undirected** graphs.
The **update rule is the same**, but for undirected graphs you initialize both directions of each edge with the same weight.

---

👉 Do you want me to also compare **Floyd–Warshall** with **Dijkstra** (which is often used for single-source shortest paths) in directed vs. undirected graphs?
