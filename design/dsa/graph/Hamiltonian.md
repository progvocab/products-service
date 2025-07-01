### 🧠 What is a Hamiltonian Path?

A **Hamiltonian Path** is a path in a **graph** that **visits each vertex exactly once**.

If the path **starts and ends at the same vertex**, it's called a **Hamiltonian Circuit (Cycle)**.

---

### 🔁 Comparison: Hamiltonian vs Eulerian

| Feature                     | Hamiltonian                              | Eulerian                |
| --------------------------- | ---------------------------------------- | ----------------------- |
| Visits                      | Every **vertex** once                    | Every **edge** once     |
| Can repeat edges?           | ✅ Yes                                    | ❌ No                    |
| Cycle ends where it starts? | Only if it's a Hamiltonian **Cycle**     | Yes (Eulerian Circuit)  |
| Solvability                 | **NP-Complete** (no known fast solution) | Solvable in linear time |
| Use cases                   | Scheduling, route planning               | Network traversal       |

---

### 📚 Real-World Problems Solved

| Domain                  | Problem                                                                            |
| ----------------------- | ---------------------------------------------------------------------------------- |
| **Traveling Salesman**  | Find shortest route visiting all cities once (Hamiltonian Cycle with minimum cost) |
| **DNA Sequencing**      | Visit every subsequence once (Hamiltonian Path in overlap graphs)                  |
| **Job Scheduling**      | Order tasks with dependencies (like topological sort → Hamiltonian Path in DAG)    |
| **Puzzle Solving**      | Knight’s Tour, sliding puzzles (graph search visiting each square once)            |
| **Data Network Design** | Routing without repetition of vertices                                             |

---

### 🔍 Hamiltonian Path Algorithm (Backtracking)

Since the problem is **NP-Complete**, we use **backtracking** to find a path.

#### 🔧 Python Implementation

```python
def hamiltonian_path(graph, N):
    path = []

    def backtrack(v, visited):
        if len(path) == N:
            return True
        for u in graph[v]:
            if not visited[u]:
                visited[u] = True
                path.append(u)
                if backtrack(u, visited):
                    return True
                visited[u] = False
                path.pop()
        return False

    for start in range(N):
        visited = [False] * N
        path = [start]
        visited[start] = True
        if backtrack(start, visited):
            return path  # Hamiltonian Path found

    return None  # No Hamiltonian Path
```

#### 🧪 Example Usage

```python
# Graph as adjacency list
graph = {
    0: [1, 3],
    1: [0, 2],
    2: [1, 3],
    3: [0, 2]
}
N = 4
result = hamiltonian_path(graph, N)
print("Hamiltonian Path:", result)
```

Output:

```
Hamiltonian Path: [0, 1, 2, 3]
```

---

### 🧩 Key Notes

* No simple necessary/sufficient condition like Eulerian path.
* Exhaustive search (backtracking) is required.
* Hamiltonian path in **DAG** (Directed Acyclic Graph) = **topological sort**.

---

### 🔄 Variants and Related Problems

| Problem                      | Description                                             |
| ---------------------------- | ------------------------------------------------------- |
| **Hamiltonian Cycle**        | Start and end at the same vertex, visit all others once |
| **TSP (Traveling Salesman)** | Find Hamiltonian Cycle with minimum weight              |
| **Topological Sort**         | A Hamiltonian Path in a DAG                             |
| **Knight’s Tour**            | Hamiltonian path/cycle on a chessboard for knight moves |
| **Hamiltonian Completion**   | Add minimum edges to make a graph Hamiltonian           |

---

Would you like:

* A **Knight's Tour example** as a Hamiltonian problem?
* Or solve the **Traveling Salesman Problem (TSP)** using Hamiltonian Cycle with brute-force or dynamic programming?
