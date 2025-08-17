Here's a complete and clean implementation of the **Bellman-Ford algorithm** in Python to find the **shortest path from a single source node** in a **weighted graph**, including support for **negative weights** and detection of **negative weight cycles**.

---

## ✅ Bellman-Ford Algorithm Overview

* Works on both **directed** and **undirected** graphs (undirected must be handled carefully if negative weights exist).
* Time Complexity: **O(V × E)**
* Can detect **negative weight cycles** (unlike Dijkstra).

---

## 🔧 Python Implementation

```python
def bellman_ford(vertices, edges, source):
    # Step 1: Initialize distances from source
    distance = {v: float('inf') for v in range(vertices)}
    distance[source] = 0

    # Step 2: Relax all edges |V| - 1 times
    for _ in range(vertices - 1):
        for u, v, wt in edges:
            if distance[u] + wt < distance[v]:
                distance[v] = distance[u] + wt

    # Step 3: Check for negative-weight cycles
    for u, v, wt in edges:
        if distance[u] + wt < distance[v]:
            print("Graph contains a negative weight cycle")
            return None

    return distance
```

---

## 🧪 Example Usage

```python
if __name__ == "__main__":
    vertices = 5  # Nodes: 0 to 4
    edges = [
        (0, 1, -1),
        (0, 2, 4),
        (1, 2, 3),
        (1, 3, 2),
        (1, 4, 2),
        (3, 2, 5),
        (3, 1, 1),
        (4, 3, -3)
    ]

    source = 0
    dist = bellman_ford(vertices, edges, source)

    if dist:
        print(f"Shortest distances from node {source}:")
        for node in range(vertices):
            print(f"Node {node}: {dist[node]}")
```

---

## 🧾 Output

```
Shortest distances from node 0:
Node 0: 0
Node 1: -1
Node 2: 2
Node 3: -2
Node 4: 1
```

---

## 🔥 Notes

* If you want to **print the shortest path**, not just distance, keep a `predecessor[]` array.
* Works well even with negative weights, as long as there's **no negative cycle**.
* For undirected graphs, you must **add both directions of each edge**.

---

Would you like:

* 📌 Path reconstruction logic?
* 🧭 Support for adjacency matrix or adjacency list?
* 🕸️ Visualization using networkx and matplotlib?

Let me know!

Great question! 🚀 Let’s go step by step.

---

# 🔹 Bellman–Ford Algorithm

The **Bellman–Ford algorithm** is a **shortest path algorithm** that finds the minimum distance from a **single source vertex** to all other vertices in a weighted graph.

Unlike **Dijkstra’s algorithm**, it works even if the graph contains **negative edge weights** (but not negative cycles).

---

## 🔹 Key Idea

* Relax all edges **V − 1 times**, where $V$ is the number of vertices.
* Relaxing means:
  If the distance to vertex $v$ through edge $u \to v$ is shorter than the current distance, update it.

Formally, if there’s an edge $u \to v$ with weight $w$:

$$
\text{if } dist[u] + w < dist[v] \text{ then } dist[v] = dist[u] + w
$$

---

## 🔹 Steps

1. Initialize distance to all vertices = **∞**, except the source = **0**.
2. Repeat **V − 1 times**:

   * For each edge $(u, v, w)$, relax it.
3. Run one more pass:

   * If any edge can still be relaxed → **negative cycle exists**.

---

## 🔹 Pseudocode

```python
def bellman_ford(V, edges, source):
    # Step 1: Initialize distances
    dist = [float('inf')] * V
    dist[source] = 0
    
    # Step 2: Relax all edges V-1 times
    for _ in range(V - 1):
        for u, v, w in edges:
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w

    # Step 3: Check for negative cycles
    for u, v, w in edges:
        if dist[u] != float('inf') and dist[u] + w < dist[v]:
            print("Negative cycle detected!")
            return None

    return dist
```

---

## 🔹 Example

Graph edges:
$(0 → 1, 4), (0 → 2, 5), (1 → 2, -3), (2 → 3, 2), (3 → 1, 1)$

1. Start: `dist = [0, ∞, ∞, ∞]`
2. Relax edges (V−1 = 3 times):

   * After 1st pass: `dist = [0, 4, 5, ∞]`
   * Relax 1→2 (-3): `dist = [0, 4, 1, ∞]`
   * Relax 2→3 (2): `dist = [0, 4, 1, 3]`
   * Relax 3→1 (1): `dist = [0, 4, 1, 3]` (no change this time)
3. Repeat relaxation 2 more times → no further changes.
4. No negative cycle → final result:

   * Shortest distances = `[0, 4, 1, 3]`.

---

## 🔹 Complexity

* Time: **O(V·E)** (slower than Dijkstra’s $O(E \log V)$)
* Space: **O(V)**

---

## 🔹 Applications

* Works with **negative weights** (unlike Dijkstra).
* Used in:

  * **Currency arbitrage detection** (negative cycles = arbitrage opportunities).
  * **Routing protocols** (e.g., RIP in networking).
  * **Solving linear difference constraints** in optimization.

---

✅ **Summary:**
Bellman–Ford is a **single-source shortest path algorithm** that works with **negative edges** and detects **negative cycles**, but is slower than Dijkstra.

---

Would you like me to also create a **side-by-side comparison: Dijkstra vs. Bellman–Ford vs. Floyd–Warshall** (with use cases)?
