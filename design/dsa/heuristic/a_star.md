In **A\*** (A-star) search, the **heuristic** is the function that estimates the cost from the current node to the goal.

The core formula for A\* is:

$$
f(n) = g(n) + h(n)
$$

Where:

* $g(n)$ = cost from start to the current node
* $h(n)$ = heuristic — **estimated** cost from current node to the goal
* $f(n)$ = total estimated cost of the cheapest solution through node $n$

---

## 1. **What makes a good heuristic?**

A heuristic should be:

* **Admissible** → Never overestimates the true cost to reach the goal
* **Consistent (monotonic)** → For every edge $n \to n'$,

  $$
  h(n) \leq cost(n, n') + h(n')
  $$

  This guarantees optimality and that $f(n)$ is non-decreasing.

---

## 2. **Examples of Heuristics**

| Problem                              | Heuristic Example                                        | Why it works                     |   |             |    |                                 |
| ------------------------------------ | -------------------------------------------------------- | -------------------------------- | - | ----------- | -- | ------------------------------- |
| 8-puzzle / 15-puzzle                 | Number of misplaced tiles                                | Admissible but not very accurate |   |             |    |                                 |
| 8-puzzle / 15-puzzle                 | Manhattan distance (sum of distances of tiles from goal) | More accurate, still admissible  |   |             |    |                                 |
| Pathfinding on a grid (4 directions) | Manhattan distance ((                                    | x\_1 - x\_2                      | + | y\_1 - y\_2 | )) | Works when no diagonals allowed |
| Pathfinding on a grid (8 directions) | Chebyshev distance (\max(                                | x\_1 - x\_2                      | , | y\_1 - y\_2 | )) | Works when diagonals allowed    |
| Euclidean space                      | Straight-line distance                                   | Admissible for shortest path     |   |             |    |                                 |

---

## 3. **How heuristics affect performance**

* **Perfect heuristic** → Search explores only the optimal path (fastest).
* **Weak heuristic** → Search becomes more like Dijkstra’s algorithm (explores more nodes).
* **Overestimating heuristic** → May break optimality guarantee.

---

## 4. **Python Example — A\* for Grid Pathfinding**

```python
import heapq
import math

def astar(start, goal, grid):
    rows, cols = len(grid), len(grid[0])

    def heuristic(a, b):
        # Euclidean distance
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

    open_set = [(0 + heuristic(start, goal), 0, start)]
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, cost, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            neighbor = (current[0]+dx, current[1]+dy)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and grid[neighbor[0]][neighbor[1]] == 0:
                tentative_g = g_score[current] + 1
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    heapq.heappush(open_set, (tentative_g + heuristic(neighbor, goal), tentative_g, neighbor))

    return None

grid = [
    [0,0,0,0],
    [1,1,0,1],
    [0,0,0,0],
    [0,1,1,0],
    [0,0,0,0]
]

path = astar((0,0), (4,3), grid)
print("Path:", path)
```

---

If you want, I can give you a **full table of common A\* heuristics for mazes, puzzles, and real-world navigation** with their admissibility proofs. That would make picking the right heuristic much easier.
