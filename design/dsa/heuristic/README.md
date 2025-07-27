Heuristics are **problem-solving techniques** that aim for *good-enough* solutions where finding the optimal one is too expensive or impractical. They are especially useful in **AI**, **search problems**, **optimization**, and **constraint satisfaction problems**.

---

## 🔍 Common Heuristics (with Python Code & Use Cases)

| Heuristic                           | Use Case                                 | Description                                  |
| ----------------------------------- | ---------------------------------------- | -------------------------------------------- |
| **1. Manhattan Distance**           | Grid-based games, A\* pathfinding        | Sum of absolute differences (no diagonals)   |
| **2. Euclidean Distance**           | Clustering, robotics                     | Straight-line (“as-the-crow-flies”) distance |
| **3. Greedy Best First**            | Pathfinding, scheduling                  | Chooses the locally best solution            |
| **4. A* Heuristic*\*                | Shortest path in weighted graphs         | Combines actual cost + estimated cost        |
| **5. Min Conflicts**                | Constraint Satisfaction (e.g., N-Queens) | Picks minimal conflict moves                 |
| **6. Hill Climbing**                | Optimization problems                    | Always moves towards a better state          |
| **7. Simulated Annealing**          | Escaping local optima                    | Adds randomness to avoid local minima        |
| **8. Heuristic Function (Game AI)** | Chess, checkers, tic-tac-toe             | Score estimation for non-terminal state      |

---

## 🧠 Example 1: A\* Algorithm with Manhattan Heuristic (Grid Pathfinding)

```python
import heapq

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    open_set = [(0 + manhattan(start, goal), 0, start)]
    came_from = {}
    cost_so_far = {start: 0}

    while open_set:
        _, cost, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        for dx, dy in [(0,1),(1,0),(-1,0),(0,-1)]:
            nx, ny = current[0]+dx, current[1]+dy
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == 0:
                next = (nx, ny)
                new_cost = cost + 1
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + manhattan(next, goal)
                    heapq.heappush(open_set, (priority, new_cost, next))
                    came_from[next] = current
    return None
```

**Usage:**

```python
grid = [
  [0,0,0],
  [1,1,0],
  [0,0,0]
]
print(astar(grid, (0,0), (2,2)))
```

---

## 🧩 Example 2: N-Queens Solver using Min-Conflicts Heuristic

```python
import random

def min_conflicts(n, max_steps=1000):
    queens = [random.randint(0, n-1) for _ in range(n)]

    def conflicts(row, col):
        return sum(
            queens[c] == row or abs(queens[c] - row) == abs(c - col)
            for c in range(n) if c != col
        )

    for _ in range(max_steps):
        conflicted = [c for c in range(n) if conflicts(queens[c], c) > 0]
        if not conflicted:
            return queens
        col = random.choice(conflicted)
        min_row = min(range(n), key=lambda r: conflicts(r, col))
        queens[col] = min_row
    return None
```

**Usage:**

```python
print(min_conflicts(8))
```

---

## 🧗 Example 3: Hill Climbing to Maximize a Function

```python
import random

def hill_climb(f, x_start, step=0.01, max_iter=1000):
    x = x_start
    for _ in range(max_iter):
        neighbors = [x + step, x - step]
        next_x = max(neighbors, key=f)
        if f(next_x) <= f(x):
            break
        x = next_x
    return x

# Maximize -x^2 + 4x
print(hill_climb(lambda x: -x**2 + 4*x, 0))
```

---

## Summary Table

| Heuristic             | Problem Solved      | Library (if any)                           |
| --------------------- | ------------------- | ------------------------------------------ |
| Manhattan / Euclidean | A\*, clustering     | `math`, `heapq`                            |
| Min-Conflicts         | N-Queens, CSP       | `random`                                   |
| Hill Climbing         | Optimization        | -                                          |
| A\* Search            | Pathfinding         | `heapq`                                    |
| Simulated Annealing   | Global optimization | `scipy.optimize.dual_annealing` (optional) |

---

Would you like code for **Simulated Annealing**, **Travelling Salesman with heuristics**, or game-based heuristics (e.g., Tic-Tac-Toe/Minimax)?




### 🔍 **Heuristic vs Algorithm** — Key Differences

| Aspect               | **Algorithm**                                                  | **Heuristic**                                                    |
| -------------------- | -------------------------------------------------------------- | ---------------------------------------------------------------- |
| **Definition**       | A **step-by-step, well-defined procedure** to solve a problem. | A **rule-of-thumb or shortcut** to find a good-enough solution.  |
| **Goal**             | To find the **correct or optimal** solution.                   | To find a **feasible or near-optimal** solution quickly.         |
| **Accuracy**         | Guaranteed (if correctly implemented).                         | Not always accurate or optimal.                                  |
| **Deterministic?**   | Yes — same input gives same output.                            | Often **non-deterministic**.                                     |
| **Speed**            | May be slower if problem is complex.                           | Typically faster, especially in large search spaces.             |
| **Example Use Case** | Sorting, shortest path, graph traversal, etc.                  | Game AI, TSP approximations, feature selection in ML.            |
| **Examples**         | Dijkstra’s, Merge Sort, Binary Search, DFS                     | A\* Search (with heuristics), Greedy methods, Genetic Algorithms |
| **When to Use**      | When exact solution is required and performance allows.        | When **exact solution is too slow** or **unknown**.              |

---

### ✅ Example: Traveling Salesman Problem (TSP)

* **Algorithm**: Brute-force checks all permutations — guarantees shortest path but **O(n!)** time.
* **Heuristic**: Nearest neighbor heuristic picks the closest unvisited city — fast, but **not always optimal**.

---

### ✅ Example: Pathfinding

* **Dijkstra’s Algorithm**: Guaranteed shortest path (no heuristic).
* **A\***: Uses a **heuristic function (like Euclidean distance)** to guess the best path — faster, but relies on a good heuristic.

---

### 📌 Summary

> **Algorithm** = Precise, complete recipe
> **Heuristic** = Informed guess, trade-off between speed and accuracy

Would you like real-world code examples comparing both for the same problem?
