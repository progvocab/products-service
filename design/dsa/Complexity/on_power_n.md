### **O(Nⁿ) Complexity Algorithms**  
O(Nⁿ) algorithms grow **exponentially**, meaning **for N = 10 and n = 3, it results in 1000 operations!** These are often seen in:  

✅ **Brute Force Recursive Algorithms**  
✅ **Backtracking & DFS for Exponential Problems**  
✅ **Generating All Permutations, Subsets, or Partitions**  
✅ **Exponential DP Problems (like TSP, Bitmask DP)**  

---

## **1️⃣ Generating All Permutations (O(Nⁿ))**  
### **🔴 Brute Force Approach**
Generating **all possible sequences of length N using N elements**.

```python
from itertools import product

def generate_sequences(n, choices):
    return list(product(choices, repeat=n))  # O(Nⁿ)

print(generate_sequences(3, [0, 1]))  
# Output: [(0,0,0), (0,0,1), (0,1,0), ..., (1,1,1)] (2³ = 8 combinations)
```

- **Why O(Nⁿ)?**  
  - Each position has **N** choices, and there are **N** positions → **Nⁿ sequences**.  

---

## **2️⃣ Backtracking for Word Generation (O(Nⁿ))**  
Generating all possible **words of length N using N letters**.

```python
def generate_words(letters, word, length):
    if length == 0:
        print(word)
        return
    for letter in letters:
        generate_words(letters, word + letter, length - 1)

generate_words("abc", "", 3)  # Generates all 3-letter words (3³ = 27)
```

- **Why O(Nⁿ)?**  
  - At each step, we choose from **N** letters, for **N** positions → **O(Nⁿ)**.  

---

## **3️⃣ Bitmask DP for Traveling Salesman Problem (O(N² * 2ⁿ))**  
### **🔴 Brute Force (O(N!))**
A naive approach to solving **Traveling Salesman Problem (TSP)**.

```python
from itertools import permutations

def tsp_brute_force(graph):
    N = len(graph)
    min_cost = float('inf')
    
    for perm in permutations(range(1, N)):  
        cost = graph[0][perm[0]]  # Start from node 0
        for i in range(len(perm) - 1):
            cost += graph[perm[i]][perm[i+1]]
        cost += graph[perm[-1]][0]  # Return to start
        min_cost = min(min_cost, cost)

    return min_cost

graph = [[0, 10, 15, 20],
         [10, 0, 35, 25],
         [15, 35, 0, 30],
         [20, 25, 30, 0]]

print(tsp_brute_force(graph))  # Too slow for large N!
```

- **Why O(N!)?**  
  - We **try all (N-1)! permutations** of cities, resulting in factorial time.  

### **✅ Optimized Using Bitmask DP (O(N² * 2ⁿ))**
Instead of checking **all permutations**, use **Dynamic Programming with Bitmasking**.

```python
from functools import lru_cache

def tsp_dp(graph):
    N = len(graph)

    @lru_cache(None)
    def dp(mask, pos):
        if mask == (1 << N) - 1:  # All cities visited
            return graph[pos][0]  # Return to start

        min_cost = float('inf')
        for nxt in range(N):
            if not (mask & (1 << nxt)):  # If city not visited
                min_cost = min(min_cost, graph[pos][nxt] + dp(mask | (1 << nxt), nxt))

        return min_cost

    return dp(1, 0)  # Start from city 0 with only city 0 visited

print(tsp_dp(graph))  # Much faster!
```

- **Why O(N² * 2ⁿ)?**  
  - Instead of **(N-1)!** permutations, we only process **N * 2ⁿ** states.

---

## **4️⃣ Generating All Subsets (O(2ⁿ))**
Generating all possible **subsets of N elements**.

```python
def generate_subsets(nums):
    subsets = []
    
    def backtrack(index, path):
        if index == len(nums):
            subsets.append(path[:])
            return
        backtrack(index + 1, path)  # Exclude element
        backtrack(index + 1, path + [nums[index]])  # Include element

    backtrack(0, [])
    return subsets

print(generate_subsets([1, 2, 3]))  
# Output: [[], [3], [2], [2,3], [1], [1,3], [1,2], [1,2,3]] (2³ = 8 subsets)
```

- **Why O(2ⁿ)?**  
  - Each element **either appears or doesn’t**, leading to **2ⁿ subsets**.

---

## **5️⃣ Chessboard Knight Moves with Backtracking (O(Nⁿ))**
A knight tries **all possible moves on an NxN board**.

```python
def knight_moves(n, x, y, steps):
    if steps == 0:
        return 1
    count = 0
    for dx, dy in [(-2, -1), (-2, 1), (2, -1), (2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2)]:
        if 0 <= x + dx < n and 0 <= y + dy < n:
            count += knight_moves(n, x + dx, y + dy, steps - 1)
    return count

print(knight_moves(8, 0, 0, 3))  
# The knight moves recursively O(Nⁿ) times!
```

- **Why O(Nⁿ)?**  
  - The knight has **up to 8 moves per step**, and explores **N depth**, leading to **O(Nⁿ)**.  

---

## **Final Comparison Table**
| **Problem** | **Algorithm** | **Time Complexity** |
|------------|--------------|--------------------|
| **Generating All Sequences** | Backtracking | **O(Nⁿ)** |
| **Word Generation** | Recursive DFS | **O(Nⁿ)** |
| **Traveling Salesman Problem** | Brute Force | **O(N!)** |
| **TSP Optimized** | Bitmask DP | **O(N² * 2ⁿ)** |
| **Generating Subsets** | Recursive Backtracking | **O(2ⁿ)** |
| **Knight Moves** | Recursive DFS | **O(Nⁿ)** |

Do you want optimizations for O(Nⁿ) problems? 🚀