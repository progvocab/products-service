### 📘 Theory of Backtracking

**Backtracking** is a general algorithmic technique for solving **recursive problems** by trying to build a solution **incrementally**, one piece at a time, and **removing solutions that fail to satisfy the constraints of the problem (backtrack)**.

---

### 🔁 Core Idea

At each decision point:

1. **Choose** a possible option.
2. **Explore** further with this option (recursively).
3. **Backtrack** — if it leads to an invalid state or doesn’t yield a full solution, undo the choice and try the next.

It is essentially **Depth-First Search (DFS)** with **state reversal**.

---

### 🧩 Structure of a Backtracking Algorithm

```python
def backtrack(state):
    if is_goal(state):
        record_solution(state)
        return
    
    for option in possible_options(state):
        if is_valid(option, state):
            make_move(option, state)
            backtrack(state)
            undo_move(option, state)  # ← this is the "backtrack"
```

---

### ✅ Problems Suited for Backtracking

* Combinatorial problems with constraints
* Problems requiring exploration of all possible states

---

### 🧠 Key Properties

| Property       | Description                                  |
| -------------- | -------------------------------------------- |
| Recursive      | Solves smaller instances recursively         |
| Pruning        | Discards infeasible solutions early          |
| Search Space   | Explores all possible options systematically |
| Backtrack Step | Reverses decision when hitting a dead-end    |

---

### 🧮 Classic Examples

| Problem                | Backtracking Approach                             |
| ---------------------- | ------------------------------------------------- |
| N-Queens Problem       | Place queens row by row and backtrack on conflict |
| Sudoku Solver          | Fill empty cells while obeying constraints        |
| Subset/Combination Sum | Try including/excluding each element              |
| Permutations           | Generate all permutations by swapping or choosing |
| Word Search            | Explore grid DFS and backtrack on wrong paths     |

---

### ⏱️ Time Complexity

Varies depending on the branching factor and depth. Typically exponential:

* Permutations of `n`: **O(n!)**
* Combinations of `n` choose `k`: **O(2^n)** worst case

---

Would you like a diagram showing the recursion tree for backtracking, like with N-Queens or subsets?
