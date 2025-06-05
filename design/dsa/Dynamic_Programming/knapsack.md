The **Knapsack Problem** is a classic optimization problem. Let's focus on solving the **0/1 Knapsack problem** (where each item can be picked **once or not at all**) using the **best algorithmic approach** for most cases: **Dynamic Programming (DP)**.

---

## 🧩 Problem Definition

Given:

* A list of `n` items, each with:

  * `value[i]`: value of the item
  * `weight[i]`: weight of the item
* A maximum capacity `W` of the knapsack

Find:

> The **maximum total value** you can fit into the knapsack without exceeding the capacity.

---

## ✅ Best Algorithm: Dynamic Programming (Bottom-Up)

### 💡 Time Complexity: `O(n * W)`

* Where `n` = number of items
* `W` = knapsack capacity

---

## 🐍 Python Implementation (Bottom-Up DP)

```python
def knapsack(values, weights, capacity):
    n = len(values)
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]

    # Build table dp[][] in bottom up manner
    for i in range(1, n + 1):
        for w in range(0, capacity + 1):
            if weights[i - 1] <= w:
                # Max of: include item i-1 or exclude it
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])
            else:
                # Cannot include item i-1
                dp[i][w] = dp[i - 1][w]

    return dp[n][capacity]

# 🧪 Example usage:
values = [60, 100, 120]
weights = [10, 20, 30]
capacity = 50
print(knapsack(values, weights, capacity))  # Output: 220
```

---

## 🧠 Optional Optimization: Space-Efficient DP

If you want to optimize space from `O(n*W)` to `O(W)`, use a **1D DP array** and iterate backwards:

```python
def knapsack_optimized(values, weights, capacity):
    n = len(values)
    dp = [0] * (capacity + 1)

    for i in range(n):
        for w in range(capacity, weights[i] - 1, -1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])

    return dp[capacity]
```

---

## 📌 Summary

| Approach            | Time Complexity | Space Complexity | Use When...                          |
| ------------------- | --------------- | ---------------- | ------------------------------------ |
| Bottom-Up DP (2D)   | `O(n * W)`      | `O(n * W)`       | Best general purpose method          |
| Space-Optimized DP  | `O(n * W)`      | `O(W)`           | When memory is a constraint          |
| Recursive with Memo | `O(n * W)`      | `O(n * W)`       | More intuitive, but slower in Python |

---

Let me know if you'd like:

* The **Fractional Knapsack** (solved via greedy)
* To **trace the items included** in the solution
* A **recursive memoized version** for clarity
