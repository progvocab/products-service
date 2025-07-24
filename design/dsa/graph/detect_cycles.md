Detecting cycles in a graph depends on whether the graph is **directed** or **undirected**. Here's how you can detect cycles in both types using Python.

---

## 📌 Cycle Detection in **Directed Graph** using DFS

### ✅ Idea:

Maintain a `visited` set and a `recursion stack` (call stack) to track nodes in the current path. If we visit a node that's already in the recursion stack, we have a cycle.

### 🔧 Code:

```python
def has_cycle_directed(graph):
    visited = set()
    rec_stack = set()

    def dfs(v):
        visited.add(v)
        rec_stack.add(v)
        for neighbor in graph.get(v, []):
            if neighbor not in visited:
                if dfs(neighbor):
                    return True
            elif neighbor in rec_stack:
                return True
        rec_stack.remove(v)
        return False

    for node in graph:
        if node not in visited:
            if dfs(node):
                return True
    return False

# 🔍 Example:
graph = {
    'A': ['B'],
    'B': ['C'],
    'C': ['A'],  # cycle: A -> B -> C -> A
    'D': ['E'],
    'E': []
}
print(has_cycle_directed(graph))  # Output: True
```

---

## 📌 Cycle Detection in **Undirected Graph** using DFS

### ✅ Idea:

Use DFS and keep track of the parent node. If a visited neighbor is not the parent, we found a cycle.

### 🔧 Code:

```python
def has_cycle_undirected(graph):
    visited = set()

    def dfs(v, parent):
        visited.add(v)
        for neighbor in graph.get(v, []):
            if neighbor not in visited:
                if dfs(neighbor, v):
                    return True
            elif neighbor != parent:
                return True
        return False

    for node in graph:
        if node not in visited:
            if dfs(node, None):
                return True
    return False

# 🔍 Example:
graph = {
    0: [1],
    1: [0, 2],
    2: [1, 3],
    3: [2, 0]  # cycle: 0-1-2-3-0
}
print(has_cycle_undirected(graph))  # Output: True
```

---

## 🧠 Notes:

* For **undirected graphs**, you can also use **Union-Find (Disjoint Set)** for cycle detection.
* For **directed graphs**, you can also use **Kahn’s Algorithm** (Topological Sort), which will fail if a cycle exists.

Would you like the Union-Find version too?



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