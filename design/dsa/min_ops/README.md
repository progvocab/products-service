 “minimum number of operations” (MNO) is one of the **most common problem patterns in competitive programming**.
They appear in **arrays, strings, number theory, DP, and graphs**.

Let’s go category by category with **common questions + intuition + techniques**:

---

# 🔹 1. Array & Sorting Problems

| Problem                       | Goal                           | Typical Techniques                 |
| ----------------------------- | ------------------------------ | ---------------------------------- |
| **Minimum swaps to sort**     | Sort array using min swaps     | Cycle decomposition of permutation |
| **Make all elements equal**   | Min ops to equalize array      | Median, prefix sums                |
| **Reduce array to zero**      | Min ops (subtract/halve)       | Greedy + bitwise                   |
| **Rotate/Reverse operations** | Sort using rotations/reversals | Prefix reversal, rotations, greedy |
| **Inversion reduction**       | Reduce inversions to zero      | Merge sort, BIT, Segment tree      |

✅ Example:

> Given an array, find min number of swaps required to sort it.
> Solution → build permutation cycles → sum over (len(cycle)-1).

---

# 🔹 2. String Problems

| Problem                           | Goal                          | Typical Techniques           |
| --------------------------------- | ----------------------------- | ---------------------------- |
| **Convert one string to another** | Min insert/delete/replace     | Edit distance (DP)           |
| **Make string palindrome**        | Min insertions/deletions      | DP (LCS with reverse string) |
| **Anagram transform**             | Min ops to convert s→t        | Frequency count difference   |
| **Remove subsequences**           | Min deletions for subsequence | Greedy + DP                  |

✅ Example:

> “Minimum insertions to make a string palindrome”
> Solution → `n - LPS(s)` where `LPS = Longest Palindromic Subsequence`.

---

# 🔹 3. Number Theory Problems

| Problem                  | Goal                      | Techniques                      |
| ------------------------ | ------------------------- | ------------------------------- |
| **Reduce GCD/LCM**       | Min ops to make gcd=1     | Change elements, gcd properties |
| **Make array divisible** | Min ops to divisible by k | Modulo counts                   |
| **Reduce number to 1**   | Min ops divide/subtract   | DP, greedy                      |
| **Prime factor removal** | Min ops to remove factors | Factorization                   |

✅ Example:

> Given array, min ops to make gcd=1.
> Solution → replace one element with 1 (if gcd>1).

---

# 🔹 4. DP / Greedy Problems

| Problem                 | Goal                           | Techniques              |
| ----------------------- | ------------------------------ | ----------------------- |
| **Coin change**         | Min coins to reach sum         | DP (unbounded knapsack) |
| **Jump game**           | Min jumps to reach end         | Greedy/DP               |
| **Frog jump (AtCoder)** | Min cost path                  | DP with transitions     |
| **Delete and earn**     | Maximize points with deletions | DP with frequency array |

✅ Example:

> “Minimum number of jumps to reach end”
> Solution → greedy O(n): keep track of max reach in current step.

---

# 🔹 5. Graph Problems

| Problem                      | Goal                      | Techniques                         |
| ---------------------------- | ------------------------- | ---------------------------------- |
| **Minimum edges to connect** | Make graph connected      | DSU / BFS                          |
| **Minimum operations BFS**   | Transform x→y             | BFS (shortest path in state space) |
| **Topological sorting ops**  | Min changes for DAG order | Kahn’s algorithm                   |

✅ Example:

> “Minimum operations to convert integer x → y (allowed: +1, \*2, -1)”
> Solution → BFS on state space.

---

# 🔹 6. Matrix / Grid Problems

| Problem                        | Goal                | Techniques       |
| ------------------------------ | ------------------- | ---------------- |
| **Min steps in grid**          | Reach target cell   | BFS              |
| **Min flips in binary matrix** | Make all 0’s or 1’s | Prefix sums, BFS |
| **Flood fill ops**             | Transform grid      | BFS/DFS          |

---

# 🔹 7. Bit Manipulation Problems

| Problem                      | Goal                      | Techniques     |
| ---------------------------- | ------------------------- | -------------- |
| **Make all numbers zero**    | Min xor/bit ops           | Count set bits |
| **Equalize arrays with XOR** | Min ops to match          | Greedy         |
| **Flip bits**                | Min flips to reach target | Prefix XOR     |

✅ Example:

> “Minimum flips to make a binary string alternating.”
> Solution → Compare with both patterns (`0101...`, `1010...`).

---

# 🔹 8. Competitive Programming Classics

1. **Minimum swaps to balance parentheses**
2. **Minimum operations to make array strictly increasing**
3. **Minimum deletions to make string good (LC 1647)**
4. **Minimum time to finish tasks with cooldown**
5. **Minimum number of operations to make array sum divisible by k**

---

## 🔹 Techniques Summary

* **Cycle decomposition** → swaps sorting
* **Dynamic Programming** → edit distance, palindromes, coin change
* **Greedy** → jump game, flips, gcd operations
* **Graph BFS** → shortest operation sequence
* **Segment Tree / BIT** → inversion counts
* **Math tricks** → gcd, lcm, modular arithmetic

---

👉 Would you like me to **pick 5 of the most common problems** (arrays, strings, DP, graph, math) and show you **Python implementations** with explanations?
