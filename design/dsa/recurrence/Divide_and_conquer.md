 Let’s dive into **divide and conquer recurrences**, because they show up *all the time* in competitive programming and algorithm analysis.

---

# 🔹 What is a Divide & Conquer Recurrence?

Divide-and-conquer algorithms work by:

1. **Divide** → Break the problem of size $n$ into smaller subproblems.
2. **Conquer** → Solve each subproblem (often recursively).
3. **Combine** → Merge the results.

This naturally leads to a **recurrence relation** for runtime.

---

# 🔹 General Form

A divide-and-conquer recurrence typically looks like:

$$
T(n) = a \, T\!\left(\frac{n}{b}\right) + f(n)
$$

Where:

* $a$ = number of subproblems created.
* $n/b$ = size of each subproblem.
* $f(n)$ = cost of dividing + combining.

---

# 🔹 Examples

### 1. Merge Sort

* Divide into 2 subproblems of size $n/2$.
* Combine (merge step) takes $O(n)$.

$$
T(n) = 2T(n/2) + O(n)
$$

Solution (via Master Theorem):

$$
T(n) = O(n \log n)
$$

---

### 2. Binary Search

* Divide into 1 subproblem of size $n/2$.
* Combine takes $O(1)$.

$$
T(n) = T(n/2) + O(1)
$$

Solution:

$$
T(n) = O(\log n)
$$

---

### 3. Strassen’s Matrix Multiplication

* Divide matrix into 7 subproblems of size $n/2$.
* Combine cost $O(n^2)$.

$$
T(n) = 7T(n/2) + O(n^2)
$$

Solution:

$$
T(n) = O(n^{\log_2 7}) \approx O(n^{2.81})
$$

---

# 🔹 How to Solve These Recurrences?

👉 Two main ways in competitive programming / algorithm analysis:

### 1. **Recursion Tree Method**

Expand the recurrence level by level until reaching the base case.

Example:

$$
T(n) = 2T(n/2) + n
$$

* Level 0: $n$
* Level 1: $2 \times (n/2)$
* Level 2: $4 \times (n/4)$
* … Each level costs $n$.
* Depth = $\log n$.

Total = $n \log n$.

---

### 2. **Master Theorem**

For recurrences of the form:

$$
T(n) = aT(n/b) + f(n)
$$

Define:

$$
n^{\log_b a}
$$

Then:

1. If $f(n) = O(n^{\log_b a - \epsilon})$, then $T(n) = \Theta(n^{\log_b a})$.
2. If $f(n) = \Theta(n^{\log_b a})$, then $T(n) = \Theta(n^{\log_b a} \log n)$.
3. If $f(n) = \Omega(n^{\log_b a + \epsilon})$, and regularity condition holds, then $T(n) = \Theta(f(n))$.

---

# 🔹 Python Example (Recursion Simulation)

```python
def mergesort_cost(n):
    if n <= 1:
        return 1
    return 2 * mergesort_cost(n // 2) + n

print(mergesort_cost(8))  # simulating recurrence expansion
```

---

# 🔹 Competitive Programming Insight

* **Merge Sort, Quick Sort, FFT** → $T(n) = 2T(n/2) + f(n)$.
* **Binary Search, Tree Traversal** → $T(n) = T(n/2) + O(1)$.
* **Matrix Multiplication** → $T(n) = aT(n/b) + f(n)$.
* **Segment Trees / Divide & Conquer DP** → often analyzed with Master Theorem.

---

👉 Do you want me to **derive the solution to a sample recurrence using the Master Theorem step by step** (like $T(n) = 3T(n/2) + n$) so you can see the mechanics in action?
