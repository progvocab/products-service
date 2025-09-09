 in **competitive programming**, trails, paths, circuits, and cycles often appear in **graph theory problems**. Let me break it down into **typical question types**, explain them, and show **Python solutions** (using DFS/BFS without heavy libraries like `networkx`, since CP expects lightweight code).

---

# 🔹 1. Path Existence

**Question:**
Given a directed graph and two vertices `u` and `v`, determine if there exists a path from `u` to `v`.

✅ **Key idea:** DFS or BFS.

```python
from collections import defaultdict, deque

def path_exists(n, edges, start, end):
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
    
    visited = set()
    q = deque([start])
    
    while q:
        node = q.popleft()
        if node == end:
            return True
        for nei in graph[node]:
            if nei not in visited:
                visited.add(nei)
                q.append(nei)
    return False

# Example
n = 4
edges = [(0,1),(1,2),(2,0),(2,3)]
print(path_exists(n, edges, 0, 3))  # ✅ True
print(path_exists(n, edges, 3, 0))  # ❌ False
```

---

# 🔹 2. Detecting Cycles

**Question:**
Does a directed graph contain a cycle?

✅ **Key idea:** DFS with recursion stack (or Kahn’s topological sort).

```python
def has_cycle(n, edges):
    graph = [[] for _ in range(n)]
    for u, v in edges:
        graph[u].append(v)

    visited = [0] * n  # 0=unvisited, 1=visiting, 2=done

    def dfs(u):
        if visited[u] == 1:
            return True  # cycle
        if visited[u] == 2:
            return False
        visited[u] = 1
        for v in graph[u]:
            if dfs(v):
                return True
        visited[u] = 2
        return False

    for i in range(n):
        if visited[i] == 0 and dfs(i):
            return True
    return False

# Example
n = 4
edges = [(0,1),(1,2),(2,0),(2,3)]
print(has_cycle(n, edges))  # ✅ True
```

---

# 🔹 3. Find All Simple Paths

**Question:**
List all **simple paths** from `u` to `v`.

✅ **Key idea:** DFS backtracking.

```python
def all_simple_paths(n, edges, start, end):
    graph = [[] for _ in range(n)]
    for u, v in edges:
        graph[u].append(v)

    result, path = [], []

    def dfs(u):
        path.append(u)
        if u == end:
            result.append(path[:])
        else:
            for v in graph[u]:
                if v not in path:  # prevent revisiting
                    dfs(v)
        path.pop()

    dfs(start)
    return result

# Example
n = 4
edges = [(0,1),(1,2),(2,3),(0,2)]
print(all_simple_paths(n, edges, 0, 3))
# ✅ [[0, 1, 2, 3], [0, 2, 3]]
```

---

# 🔹 4. Eulerian Trail / Circuit (Directed Trail Covering All Edges)

**Question:**
Does the graph contain an **Eulerian trail** (visit every edge exactly once)?

* Eulerian **circuit** exists if: every vertex has indegree = outdegree, and the graph is strongly connected.
* Eulerian **trail** exists if: at most one vertex has (outdegree−indegree)=1, and at most one has (indegree−outdegree)=1.

✅ **Key idea:** Check degree conditions.

```python
from collections import defaultdict

def has_eulerian_trail_or_circuit(edges):
    indeg, outdeg = defaultdict(int), defaultdict(int)
    nodes = set()
    
    for u, v in edges:
        outdeg[u] += 1
        indeg[v] += 1
        nodes.update([u,v])

    start_nodes = end_nodes = 0
    for node in nodes:
        if outdeg[node] - indeg[node] == 1:
            start_nodes += 1
        elif indeg[node] - outdeg[node] == 1:
            end_nodes += 1
        elif indeg[node] != outdeg[node]:
            return "No Eulerian trail/circuit"

    if start_nodes == 0 and end_nodes == 0:
        return "Eulerian Circuit exists"
    elif start_nodes == 1 and end_nodes == 1:
        return "Eulerian Trail exists"
    else:
        return "No Eulerian trail/circuit"

# Example
edges = [(0,1),(1,2),(2,0)]
print(has_eulerian_trail_or_circuit(edges))  # Eulerian Circuit exists
```

---

# 🔹 5. Hamiltonian Path / Cycle

**Question:**
Does there exist a path that visits each vertex exactly once? (Hamiltonian path/cycle)

✅ Hard (NP-complete). Often asked for small graphs (try all permutations / backtracking).

```python
import itertools

def has_hamiltonian_path(n, edges):
    graph = [[] for _ in range(n)]
    for u, v in edges:
        graph[u].append(v)

    for perm in itertools.permutations(range(n)):
        valid = True
        for i in range(n-1):
            if perm[i+1] not in graph[perm[i]]:
                valid = False
                break
        if valid:
            return True
    return False

# Example
n = 4
edges = [(0,1),(1,2),(2,3),(0,2),(2,1)]
print(has_hamiltonian_path(n, edges))  # ✅ True
```

---

# 🔹 Summary of Common CP Problems

| Problem Type           | Concept (Directed) | Typical Solution               |
| ---------------------- | ------------------ | ------------------------------ |
| Path existence         | Path               | BFS / DFS                      |
| Cycle detection        | Cycle              | DFS recursion / Kahn’s         |
| All simple paths       | Path               | DFS backtracking               |
| Eulerian trail/circuit | Trail/Circuit      | Degree counting + connectivity |
| Hamiltonian path/cycle | Path/Cycle         | Backtracking / NP-complete     |

---

👉 Do you want me to **bundle these into one Python module** with reusable functions (like `Graph.is_path()`, `Graph.has_cycle()`, `Graph.eulerian_type()` etc.), so you can directly use it in contests?
