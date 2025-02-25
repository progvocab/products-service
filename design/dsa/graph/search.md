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