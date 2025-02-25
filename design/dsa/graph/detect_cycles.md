### **Detecting Cycles in a Graph Using DFS**  

A **cycle** in a graph occurs when there is a path from a node back to itself through different edges.  

---

## **1. Cycle Detection in an Undirected Graph**  
- Use **DFS** and track **visited nodes**.  
- If we encounter a visited node that is **not the parent**, a cycle exists.  

### **Go Implementation**
```go
package main

import "fmt"

type Graph struct {
    adj map[int][]int
}

func (g *Graph) addEdge(v, w int) {
    g.adj[v] = append(g.adj[v], w)
    g.adj[w] = append(g.adj[w], v) // Undirected graph
}

func (g *Graph) isCyclicDFS(node, parent int, visited map[int]bool) bool {
    visited[node] = true

    for _, neighbor := range g.adj[node] {
        if !visited[neighbor] {
            if g.isCyclicDFS(neighbor, node, visited) {
                return true
            }
        } else if neighbor != parent {
            return true // Cycle detected
        }
    }
    return false
}

func (g *Graph) containsCycle() bool {
    visited := make(map[int]bool)

    for node := range g.adj {
        if !visited[node] {
            if g.isCyclicDFS(node, -1, visited) {
                return true
            }
        }
    }
    return false
}

func main() {
    g := Graph{adj: make(map[int][]int)}
    g.addEdge(1, 2)
    g.addEdge(2, 3)
    g.addEdge(3, 4)
    g.addEdge(4, 2) // This edge creates a cycle

    if g.containsCycle() {
        fmt.Println("Graph contains a cycle")
    } else {
        fmt.Println("Graph does not contain a cycle")
    }
}
```
### **Output**
```
Graph contains a cycle
```

---

## **2. Cycle Detection in a Directed Graph**  
- In a **directed graph**, we use **recursion stack** to track nodes in the current DFS path.  
- If a node is visited **again within the recursion stack**, a cycle exists.  

### **Go Implementation**
```go
func (g *Graph) isCyclicDirected(node int, visited, recStack map[int]bool) bool {
    visited[node] = true
    recStack[node] = true

    for _, neighbor := range g.adj[node] {
        if !visited[neighbor] {
            if g.isCyclicDirected(neighbor, visited, recStack) {
                return true
            }
        } else if recStack[neighbor] {
            return true // Cycle detected
        }
    }

    recStack[node] = false // Remove from recursion stack
    return false
}

func (g *Graph) containsCycleDirected() bool {
    visited := make(map[int]bool)
    recStack := make(map[int]bool)

    for node := range g.adj {
        if !visited[node] {
            if g.isCyclicDirected(node, visited, recStack) {
                return true
            }
        }
    }
    return false
}

func main() {
    g := Graph{adj: make(map[int][]int)}
    g.addEdge(1, 2)
    g.addEdge(2, 3)
    g.addEdge(3, 4)
    g.addEdge(4, 2) // Cycle in directed graph

    if g.containsCycleDirected() {
        fmt.Println("Graph contains a cycle")
    } else {
        fmt.Println("Graph does not contain a cycle")
    }
}
```

### **Output**
```
Graph contains a cycle
```

---

## **3. Time Complexity**
- **O(V + E)** for both **undirected and directed** graphs (same as DFS).

---

### **Key Takeaways**
✅ **DFS-based cycle detection is efficient (O(V + E))**.  
✅ **For undirected graphs**, check if a visited node **is not the parent**.  
✅ **For directed graphs**, use a **recursion stack** to track cycles.  

Would you like an example using BFS for cycle detection?