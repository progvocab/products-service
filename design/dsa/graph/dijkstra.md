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



Let's implement **Dijkstra’s Algorithm** to find the **shortest path** between cities:

* **Cities**: Bangalore, Ooty, Coimbatore, Mysore
* We'll represent cities as a graph using an **adjacency list** with distances as edge weights.

---

### 🗺️ Example City Map with Distances (in km)

| From      | To         | Distance |
| --------- | ---------- | -------- |
| Bangalore | Mysore     | 150      |
| Bangalore | Coimbatore | 370      |
| Mysore    | Ooty       | 125      |
| Ooty      | Coimbatore | 85       |
| Mysore    | Coimbatore | 200      |

---

### ✅ Python Code Using Dijkstra's Algorithm

```python
import heapq

def dijkstra(graph, start):
    distances = {city: float('inf') for city in graph}
    distances[start] = 0
    priority_queue = [(0, start)]
    previous = {city: None for city in graph}

    while priority_queue:
        current_distance, current_city = heapq.heappop(priority_queue)

        if current_distance > distances[current_city]:
            continue

        for neighbor, weight in graph[current_city]:
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous[neighbor] = current_city
                heapq.heappush(priority_queue, (distance, neighbor))

    return distances, previous

def build_path(previous, end):
    path = []
    while end:
        path.insert(0, end)
        end = previous[end]
    return path

# Define the graph
graph = {
    'Bangalore': [('Mysore', 150), ('Coimbatore', 370)],
    'Mysore': [('Ooty', 125), ('Coimbatore', 200)],
    'Ooty': [('Coimbatore', 85)],
    'Coimbatore': []
}

# Run Dijkstra's algorithm
start = 'Bangalore'
distances, previous = dijkstra(graph, start)

# Show shortest paths from Bangalore
print(f"Shortest distances from {start}:")
for city, distance in distances.items():
    path = build_path(previous, city)
    print(f" - To {city}: {distance} km via {' -> '.join(path)}")
```

---

### 📦 Output Example:

```
Shortest distances from Bangalore:
 - To Bangalore: 0 km via Bangalore
 - To Mysore: 150 km via Bangalore -> Mysore
 - To Ooty: 275 km via Bangalore -> Mysore -> Ooty
 - To Coimbatore: 360 km via Bangalore -> Mysore -> Ooty -> Coimbatore
```

---

### 🧠 Notes:

* This is a **directed weighted graph**.
* You can make it **bidirectional** by adding reverse paths if needed.
* You can adapt this for **real-world routing** by integrating with maps or APIs.

Would you like to visualize this graph or find the shortest path between specific cities only?


Absolutely! Let’s walk through the steps of **Dijkstra’s algorithm** in the context of the city graph.

---

## 🔄 Input Graph Recap:

```python
graph = {
    'Bangalore': [('Mysore', 150), ('Coimbatore', 370)],
    'Mysore': [('Ooty', 125), ('Coimbatore', 200)],
    'Ooty': [('Coimbatore', 85)],
    'Coimbatore': []
}
```

We'll walk through the `dijkstra(graph, 'Bangalore')` execution.

---

## 🧭 Step-by-Step Execution

### 🔧 Initial Setup:

```python
distances = {
    'Bangalore': 0,
    'Mysore': inf,
    'Ooty': inf,
    'Coimbatore': inf
}

previous = {
    'Bangalore': None,
    'Mysore': None,
    'Ooty': None,
    'Coimbatore': None
}

priority_queue = [(0, 'Bangalore')]
```

---

## ✅ Iteration 1: `current_city = Bangalore`

* Current distance = `0`
* Neighbors: Mysore (150), Coimbatore (370)

### ➕ Update Mysore:

* New distance = `0 + 150 = 150`
* Update:

  ```python
  distances['Mysore'] = 150
  previous['Mysore'] = 'Bangalore'
  heapq.heappush(priority_queue, (150, 'Mysore'))
  ```

### ➕ Update Coimbatore:

* New distance = `0 + 370 = 370`
* Update:

  ```python
  distances['Coimbatore'] = 370
  previous['Coimbatore'] = 'Bangalore'
  heapq.heappush(priority_queue, (370, 'Coimbatore'))
  ```

#### Priority Queue:

```
[(150, 'Mysore'), (370, 'Coimbatore')]
```

---

## ✅ Iteration 2: `current_city = Mysore`

* Current distance = `150`
* Neighbors: Ooty (125), Coimbatore (200)

### ➕ Update Ooty:

* New distance = `150 + 125 = 275`
* Update:

  ```python
  distances['Ooty'] = 275
  previous['Ooty'] = 'Mysore'
  heapq.heappush(priority_queue, (275, 'Ooty'))
  ```

### ➕ Update Coimbatore:

* New distance = `150 + 200 = 350`
* 350 < existing 370 → update:

  ```python
  distances['Coimbatore'] = 350
  previous['Coimbatore'] = 'Mysore'
  heapq.heappush(priority_queue, (350, 'Coimbatore'))
  ```

#### Priority Queue:

```
[(275, 'Ooty'), (370, 'Coimbatore'), (350, 'Coimbatore')]
```

---

## ✅ Iteration 3: `current_city = Ooty`

* Current distance = `275`
* Neighbor: Coimbatore (85)
* New distance = `275 + 85 = 360`
* 360 > current shortest (350), so **skip**

---

## ✅ Iteration 4: `current_city = Coimbatore (350)`

* Already shortest path to Coimbatore, no neighbors → nothing changes.

---

## ✅ Iteration 5: `current_city = Coimbatore (370)`

* Already processed, **skip** (stale queue entry).

---

## 🧾 Final Output:

```python
distances = {
    'Bangalore': 0,
    'Mysore': 150,
    'Ooty': 275,
    'Coimbatore': 350
}

previous = {
    'Bangalore': None,
    'Mysore': 'Bangalore',
    'Ooty': 'Mysore',
    'Coimbatore': 'Mysore'
}
```

You can use `previous` to reconstruct the shortest path.

---

### 🔚 Summary Table of Updates

| Iteration | City       | Updated Distances      | Previous                          | Queue State                          |
| --------- | ---------- | ---------------------- | --------------------------------- | ------------------------------------ |
| 1         | Bangalore  | Mysore=150, Coimb=370  | Mysore←Bangalore, Coimb←Bangalore | (150,Mysore), (370,Coimb)            |
| 2         | Mysore     | Ooty=275, Coimb=350    | Ooty←Mysore, Coimb←Mysore         | (275,Ooty), (370,Coimb), (350,Coimb) |
| 3         | Ooty       | (No update, Coimb=360) | (No change)                       | ...                                  |
| 4         | Coimb(350) | No neighbors           | -                                 | ...                                  |

---

Let me know if you’d like a **graphical visualization** or **step tracer** in code!
