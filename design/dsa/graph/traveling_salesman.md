### **Traveling Salesman Problem (TSP) - Python Solutions**
The **Traveling Salesman Problem (TSP)** is a combinatorial optimization problem where a salesman must visit **N** cities, covering the shortest possible route **exactly once**, and returning to the starting city.

---

## **1️⃣ Brute Force Approach (O(N!))**
This approach tries all possible permutations and finds the shortest one.

```python
from itertools import permutations

def tsp_brute_force(graph):
    N = len(graph)
    min_cost = float('inf')

    for perm in permutations(range(1, N)):  # All city orders (except start city 0)
        cost = graph[0][perm[0]]  # Start from city 0
        for i in range(len(perm) - 1):
            cost += graph[perm[i]][perm[i+1]]
        cost += graph[perm[-1]][0]  # Return to start
        min_cost = min(min_cost, cost)

    return min_cost

# Distance Matrix (Graph)
graph = [[0, 10, 15, 20],
         [10, 0, 35, 25],
         [15, 35, 0, 30],
         [20, 25, 30, 0]]

print("Brute Force TSP:", tsp_brute_force(graph))  # Output: 80
```

### **🔴 Why Inefficient?**
- **Tries all (N-1)! routes**, making it infeasible for **N > 10**.

---

## **2️⃣ Dynamic Programming with Bitmasking (O(N² * 2ⁿ))**
**Optimized TSP using DP + Bitmasking**.

```python
from functools import lru_cache

def tsp_dp(graph):
    N = len(graph)

    @lru_cache(None)  # Memoization
    def dp(mask, pos):
        if mask == (1 << N) - 1:  # All cities visited
            return graph[pos][0]  # Return to start

        min_cost = float('inf')
        for nxt in range(N):
            if not (mask & (1 << nxt)):  # If city not visited
                min_cost = min(min_cost, graph[pos][nxt] + dp(mask | (1 << nxt), nxt))

        return min_cost

    return dp(1, 0)  # Start from city 0 with only city 0 visited

print("Optimized TSP:", tsp_dp(graph))  # Output: 80
```

### **🟢 Why Faster?**
- Uses **bitmasking** to track visited cities (`mask = binary representation`).
- **Caches results** using `lru_cache()`.
- **Avoids duplicate calculations**.

---

## **3️⃣ Approximate Solution - Nearest Neighbor (O(N²))**
A **greedy heuristic** approach (not optimal but fast).

```python
def tsp_nearest_neighbor(graph):
    N = len(graph)
    visited = set([0])
    path = [0]
    total_cost = 0

    while len(visited) < N:
        last = path[-1]
        next_city = min(((graph[last][j], j) for j in range(N) if j not in visited), key=lambda x: x[0])
        path.append(next_city[1])
        visited.add(next_city[1])
        total_cost += next_city[0]

    total_cost += graph[path[-1]][0]  # Return to start
    return total_cost, path

print("Nearest Neighbor TSP:", tsp_nearest_neighbor(graph))  # Output: (85, [0, 1, 3, 2])
```

### **🟡 Why Inaccurate?**
- **Greedy choice** may lead to suboptimal solutions.
- Can be **30% worse** than optimal.

---

## **Comparison Table**
| **Method** | **Time Complexity** | **Optimal?** | **Pros** | **Cons** |
|------------|----------------------|-------------|----------|----------|
| **Brute Force (Permutation)** | O(N!) | ✅ Yes | Finds exact solution | **Slow for N > 10** |
| **DP + Bitmasking** | O(N² * 2ⁿ) | ✅ Yes | Optimized using memoization | Still slow for **N > 20** |
| **Nearest Neighbor (Greedy)** | O(N²) | ❌ No | Fast for large N | Suboptimal routes |

---

## **Which Approach to Use?**
- **Small N (N ≤ 10):** Use **Brute Force** for exact results.
- **Medium N (N ≤ 20):** Use **DP + Bitmasking**.
- **Large N (N > 50+):** Use **Nearest Neighbor / Approximate Algorithms**.

---

Would you like **metaheuristic solutions** like **Genetic Algorithm or Simulated Annealing**? 🚀