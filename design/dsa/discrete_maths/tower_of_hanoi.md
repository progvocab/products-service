 The **Tower of Hanoi** is one of the most famous recursion problems in competitive programming and algorithms.

---

# 📖 Problem Statement

* You have **3 rods** (source, auxiliary, destination) and **n disks** of different sizes.
* Initially, all disks are stacked on the **source rod** in decreasing size (largest at bottom).
* Objective: Move all disks to the **destination rod**.
* **Rules**:

  1. Only one disk can be moved at a time.
  2. A disk can only be placed on top of a larger disk (never smaller).

---

# 🔑 Key Idea

Recursive solution:

* To move `n` disks from `source → destination` using `auxiliary`:

  1. Move `n-1` disks from `source → auxiliary`.
  2. Move the `nth` (largest) disk from `source → destination`.
  3. Move `n-1` disks from `auxiliary → destination`.

Recurrence relation for moves:

$$
T(n) = 2T(n-1) + 1
$$

Solution:

$$
T(n) = 2^n - 1
$$

---

# 🐍 Python Solution

```python
def tower_of_hanoi(n, source, destination, auxiliary, moves=None):
    if moves is None:
        moves = []
    if n == 1:
        moves.append((source, destination))
        return moves
    tower_of_hanoi(n-1, source, auxiliary, destination, moves)
    moves.append((source, destination))
    tower_of_hanoi(n-1, auxiliary, destination, source, moves)
    return moves


# Example: 3 disks
moves = tower_of_hanoi(3, 'A', 'C', 'B')
print("Number of moves:", len(moves))
for i, (src, dest) in enumerate(moves, 1):
    print(f"Step {i}: Move disk from {src} to {dest}")
```

---

# ✅ Output (for 3 disks)

```
Number of moves: 7
Step 1: Move disk from A to C
Step 2: Move disk from A to B
Step 3: Move disk from C to B
Step 4: Move disk from A to C
Step 5: Move disk from B to A
Step 6: Move disk from B to C
Step 7: Move disk from A to C
```

---

# 🔎 Complexity

* **Moves required**: `2^n - 1`
* **Time complexity**: `O(2^n)` (exponential)
* **Space complexity**: `O(n)` recursion depth

---

👉 Do you want me to also show you an **iterative version** (using bit patterns to simulate moves), which is useful when recursion is not allowed in contests?
