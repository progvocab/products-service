Here's a complete Python implementation of a **dense graph using an adjacency matrix**, along with **Dijkstra's algorithm** to find the shortest path from a source node.

---

## 📦 Module: `dense_graph.py`

```python
import heapq
import math

class DenseGraph:
    def __init__(self, num_vertices):
        self.V = num_vertices
        self.adj_matrix = [[math.inf] * num_vertices for _ in range(num_vertices)]

    def add_edge(self, u, v, weight):
        self.adj_matrix[u][v] = weight
        # For undirected graph: uncomment next line
        # self.adj_matrix[v][u] = weight

    def dijkstra(self, start):
        dist = [math.inf] * self.V
        dist[start] = 0
        visited = [False] * self.V
        min_heap = [(0, start)]  # (distance, vertex)

        while min_heap:
            d, u = heapq.heappop(min_heap)
            if visited[u]:
                continue
            visited[u] = True

            for v in range(self.V):
                if self.adj_matrix[u][v] != math.inf and not visited[v]:
                    if dist[u] + self.adj_matrix[u][v] < dist[v]:
                        dist[v] = dist[u] + self.adj_matrix[u][v]
                        heapq.heappush(min_heap, (dist[v], v))

        return dist

    def print_matrix(self):
        for row in self.adj_matrix:
            print(row)
```

---

## ✅ Example Usage

```python
from dense_graph import DenseGraph

# Create a graph with 5 vertices
g = DenseGraph(5)

# Add edges: g.add_edge(u, v, weight)
g.add_edge(0, 1, 2)
g.add_edge(0, 2, 4)
g.add_edge(1, 2, 1)
g.add_edge(1, 3, 7)
g.add_edge(2, 4, 3)
g.add_edge(3, 4, 1)

print("Adjacency Matrix:")
g.print_matrix()

start = 0
distances = g.dijkstra(start)

print(f"\nShortest distances from vertex {start}:")
for i, d in enumerate(distances):
    print(f"Vertex {i} -> {d}")
```

---

## 🧠 Output

```
Adjacency Matrix:
[inf, 2, 4, inf, inf]
[inf, inf, 1, 7, inf]
[inf, inf, inf, inf, 3]
[inf, inf, inf, inf, 1]
[inf, inf, inf, inf, inf]

Shortest distances from vertex 0:
Vertex 0 -> 0
Vertex 1 -> 2
Vertex 2 -> 3
Vertex 3 -> 9
Vertex 4 -> 6
```

---

Would you like:

* **Path reconstruction** (not just distances)?
* **Support for undirected or weighted negative graphs**?
* **Floyd-Warshall** for all-pairs shortest path?

Let me know and I can extend this module accordingly!
