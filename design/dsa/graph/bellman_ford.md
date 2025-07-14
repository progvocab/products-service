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
