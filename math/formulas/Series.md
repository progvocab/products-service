**formulas similar to the sum of the first $n$ natural numbers**.

That one is:

$$
1 + 2 + 3 + \dots + n = \frac{n(n+1)}{2}
$$

But there are many related closed-form formulas that are often used in competitive programming and math puzzles.

---

## Common Summation Formulas

| Series                           | Formula                                                                                                   |
| -------------------------------- | --------------------------------------------------------------------------------------------------------- |
| Sum of first $n$ natural numbers | $\frac{n(n+1)}{2}$                                                                                        |
| Sum of squares                   | $1^2 + 2^2 + \dots + n^2 = \frac{n(n+1)(2n+1)}{6}$                                                        |
| Sum of cubes                     | $1^3 + 2^3 + \dots + n^3 = \left(\frac{n(n+1)}{2}\right)^2$                                               |
| Sum of first $n$ odd numbers     | $1 + 3 + 5 + \dots + (2n-1) = n^2$                                                                        |
| Sum of first $n$ even numbers    | $2 + 4 + 6 + \dots + 2n = n(n+1)$                                                                         |
| Arithmetic series                | $a + (a+d) + (a+2d) + \dots + (a+(n-1)d) = \frac{n}{2}[2a + (n-1)d]$                                      |
| Geometric series                 | $a + ar + ar^2 + \dots + ar^{n-1} = \frac{a(r^n - 1)}{r-1}, \; r \neq 1$                                  |
| Harmonic series (approx.)        | $1 + \frac{1}{2} + \frac{1}{3} + \dots + \frac{1}{n} \approx \ln(n) + \gamma$ (Euler–Mascheroni constant) |

---

## Useful Derived Forms

1. **Triangular Numbers**

   $$
   T_n = \frac{n(n+1)}{2}
   $$

   (sum of first $n$ natural numbers)

2. **Square Pyramidal Numbers**

   $$
   P_n = \frac{n(n+1)(2n+1)}{6}
   $$

   (sum of squares)

3. **Tetrahedral Numbers**

   $$
   \frac{n(n+1)(n+2)}{6}
   $$

   (sum of triangular numbers)

4. **General power sum (Faulhaber’s formula)**

   * There’s a general formula using **Bernoulli numbers** for $\sum_{k=1}^n k^p$, but for small $p$ (1, 2, 3, 4) we usually remember the closed forms.

---

## Why Useful in Competitive Programming?

* They reduce loops to $O(1)$ formulas.
* Used in **prefix sums**, **subarray sums**, and **mathematical problems**.
* Example: to compute the number of pairs $(i, j)$ where $1 \leq i < j \leq n$, we directly use $n(n-1)/2$.

---

👉 Do you want me to also give you a **ready-made list up to $k^4, k^5, k^6$** formulas (often used in contests), or just keep it till cubes and basic series?
