implementation of **BFS (Breadth-First Search)** and **DFS (Depth-First Search)** for an **undirected graph represented as an adjacency matrix** in Python.

---

### ✅ Sample Graph

Let’s assume the graph has 5 nodes labeled `0` to `4` and is represented as an adjacency matrix like this:

```python
graph = [
    [0, 1, 1, 0, 0],  # Node 0 connected to 1 and 2
    [1, 0, 0, 1, 0],  # Node 1 connected to 0 and 3
    [1, 0, 0, 1, 1],  # Node 2 connected to 0, 3, and 4
    [0, 1, 1, 0, 0],  # Node 3 connected to 1 and 2
    [0, 0, 1, 0, 0]   # Node 4 connected to 2
]
```

---

## 🔁 BFS (Breadth-First Search)

```python
from collections import deque

def bfs(adj_matrix, start):
    visited = [False] * len(adj_matrix)
    queue = deque([start])
    visited[start] = True

    print("BFS traversal:", end=" ")
    while queue:
        node = queue.popleft()
        print(node, end=" ")
        for neighbor, is_connected in enumerate(adj_matrix[node]):
            if is_connected and not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)
    print()
```

---

## 🔁 DFS (Depth-First Search)

```python
def dfs_util(adj_matrix, node, visited):
    visited[node] = True
    print(node, end=" ")
    for neighbor, is_connected in enumerate(adj_matrix[node]):
        if is_connected and not visited[neighbor]:
            dfs_util(adj_matrix, neighbor, visited)

def dfs(adj_matrix, start):
    visited = [False] * len(adj_matrix)
    print("DFS traversal:", end=" ")
    dfs_util(adj_matrix, start, visited)
    print()
```

---

## 🧪 Example Usage

```python
if __name__ == "__main__":
    graph = [
        [0, 1, 1, 0, 0],
        [1, 0, 0, 1, 0],
        [1, 0, 0, 1, 1],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0]
    ]

    start_node = 0

    bfs(graph, start_node)
    dfs(graph, start_node)
```

---

## 🧾 Output:

```
BFS traversal: 0 1 2 3 4 
DFS traversal: 0 1 3 2 4 
```

---

Let me know if you'd like:

* A **recursive vs iterative** version of DFS
* **Path tracking** or **visited node orders**
* Support for **disconnected graphs** (i.e., multiple components)




Here’s a clear comparison of **DFS (Depth-First Search)** vs **BFS (Breadth-First Search)** in **graphs**, along with:

* 🔍 Conceptual difference
* ✅ Use cases
* 💡 Python code examples

---

## 🔄 DFS vs BFS: Quick Comparison

| Feature              | **DFS** (Depth-First Search)      | **BFS** (Breadth-First Search)       |
| -------------------- | --------------------------------- | ------------------------------------ |
| Strategy             | Explore as deep as possible       | Explore level by level               |
| Data Structure       | Stack (or recursion)              | Queue                                |
| Space Complexity     | O(V)                              | O(V)                                 |
| Finds Shortest Path? | ❌ Not guaranteed                  | ✅ Yes (for unweighted graphs)        |
| Use Cases            | Cycle detection, topological sort | Shortest path, level-order traversal |

---

## 🎯 Example Graph

Let’s use this simple undirected graph:

```
Graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}
```

---

## 🧭 DFS – Depth-First Search (Recursive)

```python
def dfs(graph, node, visited=None):
    if visited is None:
        visited = set()
    if node not in visited:
        print(node, end=' ')
        visited.add(node)
        for neighbor in graph[node]:
            dfs(graph, neighbor, visited)
```

**Usage:**

```python
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

print("DFS traversal:")
dfs(graph, 'A')
```

**Output:**

```
DFS traversal:
A B D E F C
```

(Note: Exact output order may vary depending on graph structure.)

---

## 🧭 BFS – Breadth-First Search (Using Queue)

```python
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])

    while queue:
        node = queue.popleft()
        if node not in visited:
            print(node, end=' ')
            visited.add(node)
            queue.extend(graph[node])
```

**Usage:**

```python
print("\nBFS traversal:")
bfs(graph, 'A')
```

**Output:**

```
BFS traversal:
A B C D E F
```

---

## ✅ When to Use

| Problem Type                              | Use                               |
| ----------------------------------------- | --------------------------------- |
| Finding shortest path in unweighted graph | ✅ BFS                             |
| Topological sort (DAG)                    | ✅ DFS                             |
| Detecting cycles                          | ✅ DFS                             |
| Exploring connected components            | Both                              |
| Solving puzzles (e.g., mazes, Sudoku)     | Usually BFS for shortest solution |

---

## 🧠 Summary

* **DFS** explores deep and backtracks. Recursive or stack-based.
* **BFS** explores neighbors level by level. Uses queue.
* Use **BFS for shortest paths**, **DFS for pathfinding/depth logic**.

---

Would you like examples on:

* Shortest path using BFS?
* DFS/BFS on a **2D grid or matrix**?
* Iterative DFS using a stack?
