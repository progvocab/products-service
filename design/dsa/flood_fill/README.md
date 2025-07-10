### 🌊 What is **Flood Fill** in Data Structures and Algorithms?

**Flood Fill** is a classic **graph traversal algorithm** applied to a **2D grid (matrix)** to fill or label connected regions starting from a source cell.

Think of it as the logic behind the **“paint bucket” tool** in image editors — click on a pixel, and the tool fills all **connected pixels** of the same color with a new color.

---

## 🧠 Problem Statement

> Given a 2D grid, a **starting point** `(sr, sc)`, and a `newColor`, **replace all connected cells** (up, down, left, right) of the **same color** as the start cell with the `newColor`.

---

## 🔁 Example

```plaintext
Input Grid:
[
 [1, 1, 0],
 [1, 0, 0],
 [1, 1, 1]
]

Start: (0, 0), New Color: 2

Output Grid:
[
 [2, 2, 0],
 [2, 0, 0],
 [2, 2, 2]
]
```

---

## 🔄 Algorithms to Implement Flood Fill

Flood fill is basically **graph traversal** (like DFS or BFS) on a grid.

### ✅ DFS (Depth-First Search) Approach

```python
def flood_fill_dfs(image, sr, sc, new_color):
    rows, cols = len(image), len(image[0])
    start_color = image[sr][sc]

    if start_color == new_color:
        return image

    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols:
            return
        if image[r][c] != start_color:
            return

        image[r][c] = new_color

        # Visit 4 directions
        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)

    dfs(sr, sc)
    return image
```

---

### ✅ BFS (Breadth-First Search) Approach

```python
from collections import deque

def flood_fill_bfs(image, sr, sc, new_color):
    rows, cols = len(image), len(image[0])
    start_color = image[sr][sc]

    if start_color == new_color:
        return image

    queue = deque([(sr, sc)])

    while queue:
        r, c = queue.popleft()
        if image[r][c] == start_color:
            image[r][c] = new_color
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    queue.append((nr, nc))

    return image
```

---

## 🕰️ Time and Space Complexity

| Metric                     | Value                                           |
| -------------------------- | ----------------------------------------------- |
| **Time Complexity**        | O(m × n) — in worst case, all cells are visited |
| **Space Complexity (DFS)** | O(m × n) — call stack (recursive)               |
| **Space Complexity (BFS)** | O(m × n) — queue                                |

---

## 🧩 Use Cases

| Use Case                 | Description                              |
| ------------------------ | ---------------------------------------- |
| **Image editing**        | Paint bucket or region coloring          |
| **Connected components** | Counting islands or lakes                |
| **Maze solving**         | Exploring all reachable paths            |
| **Game development**     | Region detection in tile maps            |
| **Simulation**           | Fire spread, infection, flood simulation |

---

## ✅ Summary

| Topic             | Value                                                   |
| ----------------- | ------------------------------------------------------- |
| Problem Type      | Grid-based traversal                                    |
| Algorithm         | DFS or BFS                                              |
| Directional Moves | Up, Down, Left, Right (4-directional)                   |
| Common In         | Image editing, island counting, pathfinding             |
| Edge Cases        | Infinite loop if color not changed or base case skipped |

---

Would you like to try a flood fill practice problem (e.g., LeetCode 733) or add diagonal connectivity (8 directions)?
