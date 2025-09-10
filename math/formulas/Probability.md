combinatorics and probability go hand in hand in competitive programming. Let’s build the **probability formulas cheat sheet** (using the combinatorics you already know).

---

# 📘 Probability Formulas for Competitive Programming

## 🔹 1. Basics

* **Probability of event $E$:**

$$
P(E) = \frac{\text{Number of favorable outcomes}}{\text{Total outcomes}}
$$

* **Range:**

$$
0 \leq P(E) \leq 1
$$

* **Complement rule:**

$$
P(E^c) = 1 - P(E)
$$

---

## 🔹 2. Addition & Multiplication Rules

* **Union of two events:**

$$
P(A \cup B) = P(A) + P(B) - P(A \cap B)
$$

* **Intersection (if independent):**

$$
P(A \cap B) = P(A) \cdot P(B)
$$

* **General conditional case:**

$$
P(A \cap B) = P(A|B) \cdot P(B) = P(B|A) \cdot P(A)
$$

---

## 🔹 3. Conditional Probability

$$
P(A|B) = \frac{P(A \cap B)}{P(B)}, \quad P(B) > 0
$$

---

## 🔹 4. Bayes’ Theorem

$$
P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}
$$

Where

$$
P(B) = \sum_i P(B|A_i) \cdot P(A_i)
$$

---

## 🔹 5. Independent Events

Two events $A$ and $B$ are **independent** if:

$$
P(A \cap B) = P(A) \cdot P(B)
$$

---

## 🔹 6. Random Variables

* **Expected value (discrete):**

$$
E[X] = \sum_{i} x_i \cdot P(X=x_i)
$$

* **Variance:**

$$
Var(X) = E[X^2] - (E[X])^2
$$

* **Standard deviation:**

$$
\sigma = \sqrt{Var(X)}
$$

---

## 🔹 7. Binomial Distribution

Number of successes in $n$ independent Bernoulli trials (probability of success $p$):

$$
P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}
$$

* Mean: $E[X] = np$
* Variance: $Var(X) = np(1-p)$

---

## 🔹 8. Geometric Distribution

First success on the $k$-th trial:

$$
P(X = k) = (1-p)^{k-1} p
$$

Mean = $1/p$

---

## 🔹 9. Poisson Distribution

Probability of $k$ events occurring in a fixed interval (rate $\lambda$):

$$
P(X = k) = \frac{e^{-\lambda} \lambda^k}{k!}
$$

Mean = Variance = $\lambda$

---

## 🔹 10. Uniform Probability

If all $n$ outcomes are equally likely:

$$
P(E) = \frac{|E|}{n}
$$

---

## 🔹 11. Expectation Linearity

For random variables $X, Y$:

$$
E[X+Y] = E[X] + E[Y] \quad (\text{even if dependent!})
$$

---

## 🔹 12. Law of Total Probability

If $A_1, A_2, \dots, A_n$ partition the sample space:

$$
P(B) = \sum_{i=1}^n P(B|A_i) \cdot P(A_i)
$$

---

# 🔑 Quick Reference Table

| Concept              | Formula                                        |                                 |                 |
| -------------------- | ---------------------------------------------- | ------------------------------- | --------------- |
| Probability of event | $P(E) = \frac{\text{favorable}}{\text{total}}$ |                                 |                 |
| Union                | $P(A \cup B) = P(A)+P(B)-P(A \cap B)$          |                                 |                 |
| Conditional          | ( P(A                                          | B) = \frac{P(A \cap B)}{P(B)} ) |                 |
| Bayes’ theorem       | ( P(A                                          | B) = \frac{P(B                  | A)P(A)}{P(B)} ) |
| Independent events   | $P(A \cap B) = P(A)P(B)$                       |                                 |                 |
| Expectation          | $E[X] = \sum x P(X=x)$                         |                                 |                 |
| Variance             | $Var(X) = E[X^2] - (E[X])^2$                   |                                 |                 |
| Binomial             | $P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}$        |                                 |                 |
| Poisson              | $P(X=k) = \frac{e^{-\lambda}\lambda^k}{k!}$    |                                 |                 |
| Geometric            | $P(X=k) = (1-p)^{k-1}p$                        |                                 |                 |

---

⚡ In contests:

* You often combine **combinatorics** (nCr, permutations) with **probability**.
* Example: “What is the probability of drawing 2 red balls from a bag of 5 red + 3 blue?”
  → Use combinations:

  $$
  P = \frac{\binom{5}{2}}{\binom{8}{2}}
  $$

---

👉 Do you want me to extend this into a **competitive programming probability toolkit** (Python functions for nCr mod prime, binomial probability, expectation, etc.) so you can plug directly into problems?
