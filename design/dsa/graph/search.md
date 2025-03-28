## **Graph in Data Structures & Algorithms (DSA)**  

A **graph** is a non-linear data structure consisting of **nodes (vertices)** connected by **edges**. Graphs are widely used in real-world applications such as **social networks, shortest path algorithms (Google Maps), recommendation systems, and network topology**.

---

## **📌 Graph Concepts**
1. **Vertex (Node)** - A point in the graph.
2. **Edge** - A connection between two vertices.
3. **Weight** (Optional) - A value assigned to an edge in weighted graphs.
4. **Directed Graph** - Edges have direction (`A → B` but not `B → A`).
5. **Undirected Graph** - Edges have no direction (`A ↔ B`).
6. **Adjacency List** - A list where each node stores its neighbors.
7. **Adjacency Matrix** - A 2D array representing connections between nodes.
8. **Connected Graph** - A graph where every vertex is reachable.
9. **Cyclic & Acyclic Graph** - A cyclic graph contains cycles; an acyclic one does not.

---

## **📌 Time & Space Complexity**
| Operation | Adjacency List | Adjacency Matrix |
|-----------|---------------|------------------|
| **Storage** | `O(V + E)` | `O(V²)` |
| **Add Edge** | `O(1)` | `O(1)` |
| **Remove Edge** | `O(E)` | `O(1)` |
| **Check Edge** | `O(E)` | `O(1)` |
| **Iterate Neighbors** | `O(V + E)` | `O(V)` |

📌 **V = Number of vertices, E = Number of edges**

- **Adjacency List is better for sparse graphs (`E << V²`)**  
- **Adjacency Matrix is better for dense graphs (`E ≈ V²`)**

---

## **📌 Graph Representation in Python**
### **1️⃣ Adjacency List (Efficient for Sparse Graphs)**
```python
class Graph:
    def __init__(self):
        self.graph = {}

    def add_edge(self, u, v):
        if u not in self.graph:
            self.graph[u] = []
        self.graph[u].append(v)

    def display(self):
        for node in self.graph:
            print(f"{node} -> {self.graph[node]}")

g = Graph()
g.add_edge(1, 2)
g.add_edge(1, 3)
g.add_edge(2, 4)
g.display()
```
**Output:**
```
1 -> [2, 3]
2 -> [4]
```

---

### **2️⃣ Adjacency Matrix (Efficient for Dense Graphs)**
```python
class GraphMatrix:
    def __init__(self, size):
        self.size = size
        self.matrix = [[0] * size for _ in range(size)]

    def add_edge(self, u, v):
        self.matrix[u][v] = 1  # For directed graph
        self.matrix[v][u] = 1  # Add this line for an undirected graph

    def display(self):
        for row in self.matrix:
            print(row)

g = GraphMatrix(4)
g.add_edge(0, 1)
g.add_edge(1, 2)
g.add_edge(2, 3)
g.display()
```
**Output:**
```
[0, 1, 0, 0]
[1, 0, 1, 0]
[0, 1, 0, 1]
[0, 0, 1, 0]
```

---

## **📌 Graph Traversal Algorithms**
### **1️⃣ Breadth-First Search (BFS)**
BFS explores level by level using a **queue** (FIFO).

```python
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])

    while queue:
        node = queue.popleft()
        if node not in visited:
            print(node, end=" ")
            visited.add(node)
            queue.extend(graph.get(node, []))

graph = {1: [2, 3], 2: [4, 5], 3: [6], 4: [], 5: [], 6: []}
bfs(graph, 1)
```
**Output:** `1 2 3 4 5 6`

📌 **Time Complexity:** `O(V + E)`, **Space Complexity:** `O(V)`

---

### **2️⃣ Depth-First Search (DFS)**
DFS explores depth-first using a **stack** (recursion).

```python
def dfs(graph, node, visited=set()):
    if node not in visited:
        print(node, end=" ")
        visited.add(node)
        for neighbor in graph.get(node, []):
            dfs(graph, neighbor, visited)

graph = {1: [2, 3], 2: [4, 5], 3: [6], 4: [], 5: [], 6: []}
dfs(graph, 1)
```
**Output:** `1 2 4 5 3 6`

📌 **Time Complexity:** `O(V + E)`, **Space Complexity:** `O(V)`

---

## **📌 Shortest Path Algorithms**
### **1️⃣ Dijkstra’s Algorithm (Single Source Shortest Path)**
Best for **weighted graphs** (e.g., road networks).

```python
import heapq

def dijkstra(graph, start):
    pq = [(0, start)]  # (distance, node)
    distances = {node: float('inf') for node in graph}
    distances[start] = 0

    while pq:
        curr_dist, node = heapq.heappop(pq)
        for neighbor, weight in graph[node]:
            distance = curr_dist + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    
    return distances

graph = {1: [(2, 1), (3, 4)], 2: [(3, 2), (4, 5)], 3: [(4, 1)], 4: []}
print(dijkstra(graph, 1))
```
**Output:** `{1: 0, 2: 1, 3: 3, 4: 4}`

📌 **Time Complexity:** `O((V + E) log V)`

---

## **📌 Use Cases of Graphs**
| **Application** | **Graph Type** |
|---------------|---------------|
| **Google Maps (Shortest Path)** | Weighted Graph |
| **Social Networks (Friendship Suggestions)** | Undirected Graph |
| **Recommendation Systems (Netflix, Amazon)** | Bipartite Graph |
| **Web Crawling (Search Engines)** | Directed Graph |
| **Airline Route Optimization** | Weighted Graph |
| **Circuit Design & Dependencies** | Directed Acyclic Graph (DAG) |

---

## **📌 Summary**
1. **Graphs** model relationships (e.g., social networks, road maps).
2. **Representation:** Adjacency List (`O(V + E)`) vs Adjacency Matrix (`O(V²)`).
3. **Traversal:** BFS (`O(V + E)`) and DFS (`O(V + E)`).
4. **Shortest Path:** Dijkstra (`O((V + E) log V)`) for weighted graphs.
5. **Use Cases:** Networking, Recommendations, Maps, Web Crawling.

Would you like help with a **specific graph problem**? 🚀


### **Graph and Depth First Search (DFS) in Graphs**  

---

## **1. What is a Graph?**  
A **graph** is a collection of **nodes (vertices)** connected by **edges**. It can be:  
- **Directed Graph** → Edges have a direction (one-way).  
- **Undirected Graph** → Edges are bidirectional.  

Example **undirected graph**:  
```
    1 -- 2
    |    |
    3 -- 4
```

---

## **2. Depth First Search (DFS) in Graphs**  
**Depth First Search (DFS)** is a graph traversal algorithm that explores as **deep as possible** before backtracking.

### **How DFS Works?**
1. Start at a node.
2. Visit the node and mark it **visited**.
3. Recursively visit **unvisited neighbors**.
4. If no unvisited neighbors remain, **backtrack**.

**Example DFS Order (Starting from 1):**  
```
1 → 2 → 4 → 3
```

---

## **3. DFS Implementation in Go (Recursive)**
```go
package main

import "fmt"

type Graph struct {
    adj map[int][]int
}

func (g *Graph) addEdge(v, w int) {
    g.adj[v] = append(g.adj[v], w)
    g.adj[w] = append(g.adj[w], v) // For an undirected graph
}

func (g *Graph) DFS(v int, visited map[int]bool) {
    visited[v] = true
    fmt.Print(v, " ")

    for _, neighbor := range g.adj[v] {
        if !visited[neighbor] {
            g.DFS(neighbor, visited)
        }
    }
}

func main() {
    g := Graph{adj: make(map[int][]int)}
    g.addEdge(1, 2)
    g.addEdge(1, 3)
    g.addEdge(2, 4)
    g.addEdge(3, 4)

    visited := make(map[int]bool)
    fmt.Print("DFS Traversal: ")
    g.DFS(1, visited) // Output: DFS Traversal: 1 2 4 3
}
```

---

## **4. DFS Using a Stack (Iterative)**
Instead of recursion, we can use a **stack**.

```go
func (g *Graph) DFSIterative(start int) {
    stack := []int{start}
    visited := make(map[int]bool)

    for len(stack) > 0 {
        node := stack[len(stack)-1] // Peek last element
        stack = stack[:len(stack)-1] // Pop from stack

        if !visited[node] {
            visited[node] = true
            fmt.Print(node, " ")

            // Add neighbors to stack
            for _, neighbor := range g.adj[node] {
                if !visited[neighbor] {
                    stack = append(stack, neighbor)
                }
            }
        }
    }
}
```

---

## **5. DFS vs BFS (Breadth-First Search)**
| Feature | **DFS** | **BFS** |
|---------|--------|--------|
| Data Structure | Stack (recursion) | Queue |
| Exploration | Deep first | Level-wise |
| Space Complexity | O(V) (recursion depth) | O(V) (queue storage) |
| Use Case | Pathfinding, cycles detection | Shortest path, connected components |

---

## **6. Time Complexity**
| Graph Type | Time Complexity |
|------------|----------------|
| **Adjacency List** | **O(V + E)** |
| **Adjacency Matrix** | **O(V²)** |

where **V** is the number of vertices and **E** is the number of edges.

---

### **Key Takeaways**
✅ **DFS explores deeper first**, then backtracks.  
✅ **Uses recursion (or stack) to track visited nodes**.  
✅ **Efficient for cycle detection, pathfinding, and topological sorting**.  

Would you like an example of DFS detecting cycles in a graph?