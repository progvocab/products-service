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



---

## 🔹 1. **Flood Fill Algorithm**

**Definition**:

* A graph/grid traversal algorithm.
* Used to determine and “fill” a connected region starting from a seed point.
* Common in **image editing** (paint-bucket tool) or **maze solving**.

**How it works**:

* Start at a cell/pixel.
* Recursively (DFS/BFS) visit all neighboring cells of the same "type".
* Change/mark them with a new value (e.g., color).

**Key Points**:

* Input: grid, starting point, target value, replacement value.
* Output: modified grid where the region is filled.
* Time complexity: $O(N \times M)$ for an $N \times M$ grid.

**Python Example**:

```python
def flood_fill(grid, x, y, new_color):
    rows, cols = len(grid), len(grid[0])
    old_color = grid[x][y]
    if old_color == new_color:
        return grid

    def dfs(i, j):
        if 0 <= i < rows and 0 <= j < cols and grid[i][j] == old_color:
            grid[i][j] = new_color
            dfs(i+1, j)
            dfs(i-1, j)
            dfs(i, j+1)
            dfs(i, j-1)

    dfs(x, y)
    return grid
```

---

## 🔹 2. **Trapping Rain Water Problem**

**Definition**:

* Classic algorithmic problem from arrays.
* Given bar heights, compute how much rainwater is trapped between them.

**How it works**:

* For each index, trapped water = `min(max_left, max_right) - height[i]` (if positive).
* Efficient solution uses two-pointer technique.

**Key Points**:

* Input: array of heights.
* Output: integer (total trapped water).
* Time complexity: $O(n)$.
* Space complexity: $O(1)$ with two pointers.

**Python Example**:

```python
def trap(height):
    left, right = 0, len(height) - 1
    left_max = right_max = 0
    water = 0

    while left < right:
        if height[left] < height[right]:
            if height[left] >= left_max:
                left_max = height[left]
            else:
                water += left_max - height[left]
            left += 1
        else:
            if height[right] >= right_max:
                right_max = height[right]
            else:
                water += right_max - height[right]
            right -= 1
    return water
```

---

## 🔹 3. **Flood Fill vs Trapping Rain Water**

| Aspect           | Flood Fill 🌊🎨                           | Trapping Rain Water 💧🏔️                   |
| ---------------- | ----------------------------------------- | ------------------------------------------- |
| **Problem Type** | Grid traversal / connected region marking | Array-based / trapping capacity calculation |
| **Domain**       | Images, games, pathfinding                | Algorithmic puzzles, water storage modeling |
| **Input**        | 2D grid, seed point, colors/values        | 1D array of heights                         |
| **Output**       | Modified grid (filled region)             | Integer (amount of trapped water)           |
| **Algorithm**    | DFS/BFS (recursion or stack/queue)        | Two pointers / dynamic programming          |
| **Use Cases**    | Paint bucket tool, map region filling     | Rainwater harvesting, histogram analysis    |

---

✅ **Summary**:

* **Flood Fill** is about *spreading/marking connected areas* in 2D.
* **Trapping Rain Water** is about *computing storage capacity* in 1D heights.

---

👉 Do you want me to also show you how **Flood Fill can be adapted to 2D Trapping Rain Water** (the harder variation where water can be trapped inside a 2D grid)?

