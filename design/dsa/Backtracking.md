### **Backtracking Algorithm**

**Backtracking** is a **recursive** algorithmic technique for solving problems incrementally, trying one solution at a time, and removing ("backtracking") those solutions that fail to meet the conditions.

---

### **Core Idea**

1. Choose an option
2. Explore it recursively
3. If it leads to a dead-end, backtrack and choose another option

It’s often used for problems that require **exploring all combinations** or **paths**, like:
- Puzzles (Sudoku, N-Queens)
- Combinations / permutations
- Path-finding
- Constraint satisfaction problems

---

### **Time and Space Complexity**

- **Time:** Exponential in nature — O(k^n) where `n` is the depth, and `k` is the number of choices at each level.
- **Space:** O(n) for recursion stack + possible space for state storage (e.g., path array)

---

### **Use Cases**

| Problem Type              | Examples                               |
|---------------------------|----------------------------------------|
| Combinatorial             | Subsets, permutations, combinations    |
| Constraints-based         | Sudoku, N-Queens                       |
| Pathfinding               | Maze problems                          |
| String problems           | Palindrome partitioning, Word Break   |

---

### **Example 1: Subsets (Power Set)**

```python
def subsets(nums):
    result = []

    def backtrack(start, path):
        result.append(path[:])
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()

    backtrack(0, [])
    return result

print(subsets([1, 2, 3]))
# Output: All subsets of [1,2,3]
```

---

### **Example 2: N-Queens Problem**

```python
def solve_n_queens(n):
    results = []

    def is_valid(board, row, col):
        for r in range(row):
            if board[r] == col or \
               abs(board[r] - col) == abs(r - row):
                return False
        return True

    def backtrack(row, board):
        if row == n:
            results.append(board[:])
            return
        for col in range(n):
            if is_valid(board, row, col):
                board[row] = col
                backtrack(row + 1, board)
                board[row] = -1

    backtrack(0, [-1] * n)
    return results

print(solve_n_queens(4))
```

---

### **Example 3: Maze Path**

```python
def maze_solver(maze, x=0, y=0, path=[]):
    if x == len(maze) - 1 and y == len(maze[0]) - 1:
        return [path + [(x, y)]]

    if x >= len(maze) or y >= len(maze[0]) or maze[x][y] == 0:
        return []

    maze[x][y] = 0  # Mark as visited
    moves = [(1, 0), (0, 1)]  # Down, Right
    result = []
    for dx, dy in moves:
        result += maze_solver(maze, x + dx, y + dy, path + [(x, y)])
    maze[x][y] = 1  # Backtrack
    return result

maze = [
    [1, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
]
print(maze_solver(maze))
```

---

### **Summary**

| Feature        | Details                                         |
|----------------|--------------------------------------------------|
| Strategy       | Recursive, Try-all-possibilities                 |
| Time Complexity| Often exponential (O(k^n))                       |
| Space Complexity| O(n) (recursion + state tracking)              |
| Use Cases      | Puzzles, Combinatorics, Pathfinding             |
| Design Pattern | Often uses **DFS + Backtracking**               |

Would you like optimized versions of any example using pruning or memoization?