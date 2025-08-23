Ah, I think you meant **Held–Karp algorithm** (sometimes misheard as *herp karb*) for the **Travelling Salesman Problem (TSP)** 🙂.
Let’s go step by step:

---

# 🔹 Travelling Salesman Problem (TSP)

* You have **n cities** and distances between every pair.
* A salesman starts at a city, visits each city **exactly once**, and returns to the starting city.
* Goal: Find the **minimum possible travel cost**.

This is an **NP-hard problem**.

---

# 🔹 Held–Karp Algorithm (Dynamic Programming for TSP)

The Held–Karp algorithm is a **Dynamic Programming (DP)** approach that solves TSP in **O(n² · 2ⁿ)** time, which is much better than the brute force **O(n!)**.

---

## 🧠 Idea

We use **bitmask DP** to represent subsets of cities.

### DP State:

$$
dp[S][j] = \text{minimum cost to reach city j, visiting all cities in set S exactly once}
$$

* $S$ = subset of visited cities (represented as a **bitmask**)
* $j$ = last visited city in subset $S$

---

### Recurrence:

$$
dp[S][j] = \min_{i \in S, i \neq j} \big( dp[S \setminus \{j\}][i] + \text{dist}[i][j] \big)
$$

* Transition: To reach city `j` with visited set `S`, we must have come from some `i` in `S`.

---

### Base Case:

$$
dp[\{0\}][0] = 0
$$

(start at city 0 with only city 0 visited)

---

### Final Answer:

$$
\min_{j \neq 0} \big( dp[\{all\_cities\}][j] + \text{dist}[j][0] \big)
$$

---

# 🔹 Example Python Implementation

```python
import math

def tsp_held_karp(dist):
    n = len(dist)
    # dp[mask][i] = min cost to visit subset mask, ending at i
    dp = [[math.inf] * n for _ in range(1 << n)]
    dp[1][0] = 0  # starting at city 0

    for mask in range(1 << n):
        for j in range(n):
            if not (mask & (1 << j)):  # city j not in mask
                continue
            # try to reach j from some i
            for i in range(n):
                if i != j and (mask & (1 << i)):
                    dp[mask][j] = min(dp[mask][j],
                                      dp[mask ^ (1 << j)][i] + dist[i][j])

    # Final: return to city 0
    full_mask = (1 << n) - 1
    return min(dp[full_mask][j] + dist[j][0] for j in range(1, n))


# Example: distance matrix for 4 cities
dist_matrix = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
]

print("Minimum TSP cost:", tsp_held_karp(dist_matrix))
```

---

## ✅ Output

```
Minimum TSP cost: 80
```

---

# 🔹 Complexity

* **Time:** $O(n^2 \cdot 2^n)$
* **Space:** $O(n \cdot 2^n)$

Much better than brute force $O(n!)$, but still exponential.

---

# 🔹 Applications of Held–Karp

* Route optimization (delivery, logistics)
* Circuit design & PCB layout
* DNA sequencing in bioinformatics
* Network routing problems

---

👉 Would you like me to also show you how **Held–Karp (exact)** compares with **approximation heuristics** like **Nearest Neighbor** and **Christofides’ algorithm** for TSP?
