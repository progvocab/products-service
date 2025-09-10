let’s build a **Recurrence & Dynamic Programming Formula Handbook** for competitive programming.
I’ll organize it into categories with the most **commonly needed recurrence relations, closed forms, and DP formulas**.

---

# 📘 Recurrences & DP Formulas for Competitive Programming

---

## 🔹 1. Linear Recurrences

* **General linear recurrence of order k**

$$
T(n) = a_1 T(n-1) + a_2 T(n-2) + \dots + a_k T(n-k)
$$

* **Characteristic polynomial method** (for closed form):

$$
x^k - a_1 x^{k-1} - a_2 x^{k-2} - \dots - a_k = 0
$$

---

### Common Examples

* **Fibonacci Sequence**

$$
F(n) = F(n-1) + F(n-2), \quad F(0)=0, F(1)=1
$$

Closed form (Binet’s formula):

$$
F(n) = \frac{1}{\sqrt{5}}\left(\phi^n - \hat{\phi}^n\right), \quad \phi=\frac{1+\sqrt{5}}{2}, \ \hat{\phi}=\frac{1-\sqrt{5}}{2}
$$

* **Tribonacci**

$$
T(n) = T(n-1) + T(n-2) + T(n-3)
$$

* **Arithmetic progression**

$$
a_n = a_{n-1} + d
$$

$$
a_n = a_1 + (n-1)d
$$

* **Geometric progression**

$$
a_n = r \cdot a_{n-1}
$$

$$
a_n = a_1 r^{n-1}
$$

---

## 🔹 2. Divide & Conquer Recurrences (Master Theorem)

$$
T(n) = a \, T\!\left(\frac{n}{b}\right) + f(n)
$$

* Case 1: If $f(n) = O(n^{\log_b a - \epsilon})$ →

$$
T(n) = \Theta(n^{\log_b a})
$$

* Case 2: If $f(n) = \Theta(n^{\log_b a}\log^k n)$ →

$$
T(n) = \Theta(n^{\log_b a}\log^{k+1} n)
$$

* Case 3: If $f(n) = \Omega(n^{\log_b a + \epsilon})$ and regularity condition holds →

$$
T(n) = \Theta(f(n))
$$

✅ Used in analyzing algorithms like Merge Sort, Karatsuba, Strassen.

---

## 🔹 3. Common DP Recurrences

### Subset/Knapsack

* **0/1 Knapsack**:

$$
dp[i][w] = \max(dp[i-1][w], \ dp[i-1][w-w_i] + v_i)
$$

* **Unbounded Knapsack**:

$$
dp[i][w] = \max(dp[i-1][w], \ dp[i][w-w_i] + v_i)
$$

### LIS (Longest Increasing Subsequence)

$$
dp[i] = 1 + \max_{j < i, \ arr[j] < arr[i]} dp[j]
$$

### Matrix Chain Multiplication

$$
dp[i][j] = \min_{k=i}^{j-1} (dp[i][k] + dp[k+1][j] + cost)
$$

### Coin Change (Count Ways)

$$
dp[n] = \sum_{c \in coins, \ n-c \ge 0} dp[n-c]
$$

---

## 🔹 4. Catalan Numbers (DP in combinatorics)

$$
C_n = \frac{1}{n+1}\binom{2n}{n}
$$

Recurrence form:

$$
C_{n+1} = \sum_{i=0}^{n} C_i \cdot C_{n-i}, \quad C_0 = 1
$$

Applications:

* Parenthesization count
* Binary tree structures
* Dyck paths
* Polygon triangulations

---

## 🔹 5. Special DP Recurrences

* **Bell Numbers** (partitions of a set):

$$
B_{n+1} = \sum_{k=0}^{n} \binom{n}{k} B_k, \quad B_0 = 1
$$

* **Stirling Numbers of 2nd Kind** (partitions into k groups):

$$
S(n,k) = k \cdot S(n-1,k) + S(n-1,k-1)
$$

* **Derangements** (permutations with no fixed points):

$$
!n = (n-1)(!(n-1) + !(n-2)), \quad !0=1,!1=0
$$

---

## 🔹 6. Matrix Exponentiation for DP Recurrences

If recurrence is linear:

$$
T(n) = a_1 T(n-1) + a_2 T(n-2) + \dots + a_k T(n-k)
$$

Then it can be expressed as:

$$
\begin{bmatrix}
T(n) \\ T(n-1) \\ \vdots \\ T(n-k+1)
\end{bmatrix}
=
M \cdot
\begin{bmatrix}
T(n-1) \\ T(n-2) \\ \vdots \\ T(n-k)
\end{bmatrix}
$$

where $M$ is the recurrence matrix.
Fast exponentiation gives $O(k^3 \log n)$ solution.

---

## 🔹 7. Generating Functions

* Used to solve recurrences like:

$$
a_n = a_{n-1} + 2a_{n-2}
$$

Generating function:

$$
A(x) = \sum_{n=0}^{\infty} a_n x^n
$$

Equation transforms recurrence → closed form.

---

✅ This covers:

* **Linear recurrences** (Fibo, Tribo, AP, GP)
* **Divide & Conquer** (Master theorem)
* **Classical DP recurrences** (Knapsack, LIS, Matrix Chain, Coin Change)
* **Combinatorial DP** (Catalan, Bell, Stirling, Derangements)
* **Advanced** (Matrix exponentiation, generating functions)

---

Would you like me to now make a **combined cheat sheet with Python templates** for each of these (like matrix exponentiation for Fibonacci, knapsack DP, Catalan DP), so you can directly plug into contest solutions?
