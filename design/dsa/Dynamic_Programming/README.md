### **Dynamic Programming (DP) Theory**

**Dynamic Programming** is a method for solving complex problems by breaking them down into simpler overlapping subproblems and storing the results of subproblems to avoid redundant computation.

---

### **Core Principles of Dynamic Programming**

1. **Optimal Substructure**:

   * A problem has an optimal substructure if an optimal solution to the problem contains optimal solutions to its subproblems.
   * Example: Shortest path in a graph, Fibonacci numbers.

2. **Overlapping Subproblems**:

   * The problem can be broken down into subproblems that are reused multiple times.
   * Unlike divide-and-conquer, where subproblems are mostly non-overlapping.

3. **Memoization (Top-Down)**:

   * Store the result of each subproblem the first time it is computed.
   * Subsequent calls reuse the stored result instead of recomputing.

4. **Tabulation (Bottom-Up)**:

   * Solve subproblems in order and store their results in a table (array or matrix).
   * Use previously computed values to build up the final solution.

---

### **Steps to Solve a DP Problem**

1. **Identify the subproblem**
2. **Formulate a recurrence relation**
3. **Decide the direction (top-down or bottom-up)**
4. **Implement memoization or tabulation**
5. **Initialize base cases**
6. **Build solution step by step**

---

### **Common Use Cases**

* Fibonacci sequence
* Knapsack problem
* Coin change
* Longest common subsequence (LCS)
* Matrix chain multiplication
* Edit distance
* Rod cutting problem

---

### **Example: Fibonacci (Tabulation)**

```python
def fibonacci(n):
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n+1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]
```

---

### **Time & Space Complexity**

* Time complexity is usually polynomial (`O(n)`, `O(n^2)`, etc.)
* Space complexity depends on how much you store. Optimizations like using only previous values reduce space.

---

Let me know if you want a visual flow or example for a specific problem like Knapsack or LCS.
