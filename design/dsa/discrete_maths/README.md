 This is the **sweet spot** where discrete math meets algorithms and competitive programming.
Discrete mathematics gives us **formulas + reasoning tools** that directly solve common CP problems.

Let’s go through the **big buckets of problems**, the math behind them, and examples.

---

# 🔹 1. **Combinatorics (Counting Problems)**

👉 Count ways to arrange, choose, or partition elements.

| Problem Type                     | Discrete Math Formula / Algorithm                           | Example                               |
| -------------------------------- | ----------------------------------------------------------- | ------------------------------------- |
| **Permutations**                 | `n!` ways to arrange `n` elements                           | Arrange `ABC` → 6 ways                |
| **Combinations**                 | `nCr = n! / (r!(n-r)!)`                                     | Choose 2 out of 5 players             |
| **Permutations with repetition** | `n^r`                                                       | Generate all passwords of length `r`  |
| **Multiset permutations**        | `n! / (n1! n2! …)`                                          | Arrange “AAB” = 3!/2! = 3             |
| **Pigeonhole principle**         | If `n+1` items in `n` boxes → at least one box has 2+ items | Birthday paradox                      |
| **Inclusion-Exclusion**          | Count total by adding, subtracting overlaps                 | Count numbers ≤ N divisible by 2 or 3 |

✅ CP Example:
"How many subsets of size k have sum divisible by m?" → combinations + modular arithmetic.

---

# 🔹 2. **Number Theory**

👉 Modular arithmetic, gcd, primes.

| Problem Type                        | Formula / Algorithm                 | Example                      |
| ----------------------------------- | ----------------------------------- | ---------------------------- |
| **GCD / LCM**                       | Euclidean Algorithm                 | gcd(48,18)=6                 |
| **Modular exponentiation**          | Fast power (binary exponentiation)  | Compute `a^b mod m`          |
| **Modular inverse**                 | Extended Euclid or Fermat’s theorem | Solve `3x ≡ 1 mod 11` → x=4  |
| **Euler’s Totient φ(n)**            | Count integers coprime to `n`       | φ(9)=6                       |
| **Chinese Remainder Theorem (CRT)** | Solve simultaneous congruences      | x ≡ 2 (mod 3), x ≡ 3 (mod 5) |
| **Fermat’s Little Theorem**         | `a^(p-1) ≡ 1 mod p`                 | Prime mod properties         |

✅ CP Example:
"Find nCr % p for large n" → factorial + modular inverse.

---

# 🔹 3. **Graph Theory (Discrete Structures)**

👉 Problems modeled as nodes + edges.

| Problem Type              | Algorithm / Formula                  | Example                          |
| ------------------------- | ------------------------------------ | -------------------------------- |
| **Shortest path**         | Dijkstra / BFS (unweighted)          | Minimum steps in a maze          |
| **Minimum Spanning Tree** | Kruskal, Prim                        | Network design                   |
| **Connected components**  | DFS / Union-Find                     | Count groups in social network   |
| **Eulerian path**         | Exists if 0 or 2 vertices odd degree | Find a path using all roads once |
| **Hamiltonian path**      | NP-hard                              | Traveling salesman problem       |
| **Graph coloring**        | Chromatic number, greedy             | Scheduling exams                 |

---

# 🔹 4. **Logic & Boolean Algebra**

👉 Used in conditions, bitmasks, digital problems.

| Problem Type            | Discrete Math Formula                 | Example             |
| ----------------------- | ------------------------------------- | ------------------- |
| **Propositional logic** | Truth tables, DeMorgan’s laws         | Simplify conditions |
| **Bitmasking**          | Subsets of `n` elements → `2^n` masks | DP on subsets       |
| **Boolean algebra**     | Simplification of logic gates         | Optimize circuit    |

✅ CP Example:
"Find subset with sum ≤ K" → enumerate subsets using bitmasks.

---

# 🔹 5. **Set Theory & Relations**

👉 Used in DP, database joins, equivalence classes.

| Problem Type             | Math Concept                    | Example                |
| ------------------------ | ------------------------------- | ---------------------- |
| **Power set**            | `2^n` subsets of set of size n  | Subset-sum problem     |
| **Equivalence relation** | Partition elements into classes | Friend circles problem |
| **Partial orders**       | Topological sorting             | Task scheduling        |

---

# 🔹 6. **Recurrence Relations**

👉 Problems that repeat structure.

| Problem Type            | Method                                                       | Example                         |
| ----------------------- | ------------------------------------------------------------ | ------------------------------- |
| **Linear recurrence**   | Solve with characteristic polynomial / matrix exponentiation | Fibonacci: F(n) = F(n-1)+F(n-2) |
| **Divide & Conquer**    | Master theorem                                               | T(n)=2T(n/2)+n → O(n log n)     |
| **Dynamic Programming** | Tabulation/memoization                                       | Knapsack, LIS                   |

---

# 🔹 7. **Probability & Statistics (Discrete)**

👉 Counting + probability theory.

| Problem Type              | Formula                      | Example                                 |
| ------------------------- | ---------------------------- | --------------------------------------- |
| **Expected value**        | `E[X] = Σ x·P(x)`            | Expected dice roll = 3.5                |
| **Binomial distribution** | `P(k) = nCk p^k (1-p)^(n-k)` | Probability of k heads in n coin tosses |
| **Markov chains**         | Transition matrices          | Random walk problems                    |

---

# 🔹 8. **Linear Algebra (Discrete Applications)**

👉 Matrices over integers/finite fields.

| Problem Type                  | Formula / Method                          | Example               |
| ----------------------------- | ----------------------------------------- | --------------------- |
| **Matrix exponentiation**     | Fast recurrence calculation               | Fibonacci in O(log n) |
| **Gaussian elimination**      | Solve linear equations                    | System of congruences |
| **Graph as adjacency matrix** | `A^k[i][j]` = number of walks of length k | Count paths in graph  |

---

# ✅ Summary

Discrete math formulas/algorithms solve **a wide range of CP problems**:

1. **Combinatorics** → counting subsets, arrangements.
2. **Number theory** → modular arithmetic, inverses, primes.
3. **Graph theory** → shortest paths, connectivity, trees.
4. **Logic & Boolean algebra** → bitmasking, optimization.
5. **Set theory & relations** → partitions, equivalence, DP states.
6. **Recurrence relations** → DP, divide & conquer, matrix exponentiation.
7. **Probability** → expected values, random processes.
8. **Linear algebra** → solving equations, matrix exponentiation.

---

👉 Do you want me to now **pick 3–4 real competitive programming problems** (like from Codeforces/LeetCode) and solve them **step by step using discrete math formulas** so you see theory → code in action?
