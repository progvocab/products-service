---

## 1. **What is "state" in DP?**

In **Dynamic Programming (DP)**, a **state** is a way to represent a **subproblem** you’re solving.
It’s the **information needed** to uniquely describe where you are in the problem.

* Example (Fibonacci):
  State = `dp[i]` = "the i-th Fibonacci number".
  Here the **state** is just the index `i`.

* Example (Knapsack):
  State = `dp[i][w]` = "the maximum value achievable using first `i` items with capacity `w`".
  Here the **state** is both the **item index** `i` and the **remaining capacity** `w`.

Think of the **state** as the "coordinates" of your problem inside a DP table.

---

## 2. **What is 1D DP?**

**1D DP = One-Dimensional Dynamic Programming.**

* Your DP table/array has **one dimension**.
* Meaning the state only depends on **one parameter** (like index, number of steps, amount, etc.).

🔹 Example: Fibonacci

```python
dp = [0] * (n+1)
dp[0] = 0
dp[1] = 1
for i in range(2, n+1):
    dp[i] = dp[i-1] + dp[i-2]
```

Here, `dp[i]` is a **1D array** → only depends on `i`.

---

## 3. **What is 2D DP?**

**2D DP = Two-Dimensional Dynamic Programming.**

* Your DP table has **two dimensions**.
* State depends on **two independent parameters**.

🔹 Example: 0/1 Knapsack

```python
dp = [[0]*(W+1) for _ in range(n+1)]
for i in range(1, n+1):
    for w in range(1, W+1):
        if weight[i-1] <= w:
            dp[i][w] = max(dp[i-1][w], value[i-1] + dp[i-1][w-weight[i-1]])
        else:
            dp[i][w] = dp[i-1][w]
```

Here, `dp[i][w]` is **2D** because we need both:

* `i` = number of items considered
* `w` = capacity

---

## 4. **Why the difference?**

* If a problem can be solved by tracking only one variable → **1D DP** is enough.
* If you need to track two variables simultaneously → you need **2D DP**.

---

✅ Quick Analogy:

* **1D DP**: like keeping a **line of boxes**, each box depends on the previous ones.
* **2D DP**: like keeping a **grid of boxes**, each box depends on others both **above** and **to the left**.

---

## 🔑 General Idea

* **1D DP** → if the state only depends on **one parameter** (or if multiple parameters can be reduced to one dimension at a time).
* **2D DP** → if the state naturally depends on **two independent parameters** (like index + sum, index + capacity, index + length, etc.).

---

## 📌 Rules of Thumb

### 1. Look at the **recurrence relation**

* If your state is `dp[i]` and only depends on a single index (`i`) or number, then **1D is enough**.
* If your state is `dp[i][j]` where you need both:

  * `i = position / element index`
  * `j = extra parameter (sum, capacity, length, remaining, etc.)`
    then you need **2D DP**.

---

### 2. Examples

#### ✅ 1D DP Examples

* **Fibonacci**
  `dp[i] = dp[i-1] + dp[i-2]`
  Only depends on index `i`.

* **Climbing Stairs (LeetCode 70)**
  `dp[i] = dp[i-1] + dp[i-2]`
  Again only depends on the number of steps.

* **House Robber (LeetCode 198)**
  `dp[i] = max(dp[i-1], dp[i-2] + nums[i])`
  Depends only on one dimension (the index).

---

#### ✅ 2D DP Examples

* **Target Sum (LeetCode 494)**
  State: `dp[i][sum]` = number of ways to reach `sum` using first `i` numbers.
  Needs both **index** and **current sum**.

* **Knapsack (0/1)**
  State: `dp[i][capacity]` = max value using first `i` items and capacity `capacity`.
  Needs both **item index** and **capacity**.

* **Edit Distance (LeetCode 72)**
  State: `dp[i][j]` = min operations to convert first `i` chars of word1 to first `j` chars of word2.
  Needs **two string indices**.

---

### 3. Space Optimization

Even if the **natural recurrence is 2D**, sometimes you can optimize to **1D**:

* In **knapsack problems**, you can reduce `dp[i][capacity]` to `dp[capacity]` if you carefully update in reverse order.
* In **Target Sum**, you can reduce `dp[i][sum]` to a 1D array by rolling over.

---

## 📌 Checklist to Identify

When solving:

1. Ask yourself: "What do I need to uniquely define the state?"

   * If it’s **just one variable** → 1D.
   * If it’s **two independent variables** → 2D.
2. Write the recurrence. If it looks like `dp[i] = ...` → 1D. If it looks like `dp[i][j] = ...` → 2D.
3. After solving, see if dimensions can be reduced (space optimization).
---
3D DP is the natural extension of 1D and 2D DP. 

---

## 1. **What is 3D DP?**

**3D DP (Three-Dimensional Dynamic Programming)** means your DP state depends on **three independent variables**.
So your DP table is shaped like a **cube** (3D array).

The **state** is usually written as:

$$
dp[i][j][k] = \text{answer to subproblem defined by } (i, j, k)
$$

---

## 2. **When do we need 3D DP?**

Whenever the problem depends on **three independent factors** simultaneously.

🔹 Common examples:

1. **Longest Common Subsequence with 3 strings**

   * `dp[i][j][k]` = LCS length of first `i` chars of string1, first `j` chars of string2, first `k` chars of string3.

2. **Knapsack with extra constraints**

   * `dp[i][w][c]` = max value using first `i` items, weight `w`, and cost `c`.

3. **Edit distance with extra dimension**

   * `dp[i][j][k]` = min operations for first `i` chars of word1, first `j` chars of word2, given at most `k` substitutions.

---

## 3. **Example: LCS of 3 strings**

Problem: Find the **longest common subsequence** among 3 strings.

```python
def lcs3(s1, s2, s3):
    n1, n2, n3 = len(s1), len(s2), len(s3)
    dp = [[[0]*(n3+1) for _ in range(n2+1)] for _ in range(n1+1)]

    for i in range(1, n1+1):
        for j in range(1, n2+1):
            for k in range(1, n3+1):
                if s1[i-1] == s2[j-1] == s3[k-1]:
                    dp[i][j][k] = dp[i-1][j-1][k-1] + 1
                else:
                    dp[i][j][k] = max(
                        dp[i-1][j][k],
                        dp[i][j-1][k],
                        dp[i][j][k-1]
                    )
    return dp[n1][n2][n3]

print(lcs3("abc", "ac", "bc"))  # Output: 1
```

Here the state is:

* `i` → prefix length of string1
* `j` → prefix length of string2
* `k` → prefix length of string3

---

## 4. **Complexity**

* If each dimension has size `N`, then:

  * **Time Complexity**: `O(N³)`
  * **Space Complexity**: `O(N³)` (can often be reduced by rolling arrays).

---

✅ Summary:

* **1D DP** → state depends on 1 variable.
* **2D DP** → state depends on 2 variables.
* **3D DP** → state depends on 3 variables (visualize as a cube of subproblems).


