Sure! Let's solve the **0/1 Knapsack problem** using a **Brute Force Bottom-Up approach** **without recursion**.

---

### 🎒 Problem Recap:

Given:

* `weights[]`: array of item weights
* `values[]`: array of item values
* `W`: capacity of the knapsack

Goal: **Maximize the value** of items put in the knapsack **without exceeding the weight limit**, where each item can be included **at most once**.

---

### ✅ Bottom-Up Brute Force Approach (Non-recursive)

We'll build a 2D DP table where:

* `dp[i][w]` = **maximum value** using the **first `i` items** with **capacity `w`**

```python
def knapsack_bottom_up(weights, values, W):
    n = len(weights)
    
    # dp[i][w] = max value for first i items with weight limit w
    dp = [[0 for _ in range(W + 1)] for _ in range(n + 1)]

    # Build the table in bottom-up fashion
    for i in range(1, n + 1):
        for w in range(W + 1):
            if weights[i - 1] <= w:
                # Choose max of including or excluding the item
                dp[i][w] = max(
                    dp[i - 1][w],  # exclude item
                    values[i - 1] + dp[i - 1][w - weights[i - 1]]  # include item
                )
            else:
                dp[i][w] = dp[i - 1][w]  # item can't be included

    return dp[n][W]

# Example usage
weights = [2, 3, 4, 5]
values = [3, 4, 5, 6]
capacity = 5

print("Max value:", knapsack_bottom_up(weights, values, capacity))
```

---

### 🧠 Time & Space Complexity

* **Time:** `O(n * W)`
* **Space:** `O(n * W)` (can be optimized to `O(W)` using 1D DP array)

---

Let me know if you’d like:

* The **selected items** to be returned,
* A **1D space-optimized** version,
* Or the **unbounded knapsack** variant.
