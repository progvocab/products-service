 **Bernoulli trial process** (the concept is **Bernoulli trials** in probability and stochastic processes). Let’s go step by step.

---

# 🔹 What is a Bernoulli Trial?

* A **Bernoulli trial** is an experiment with **only two outcomes**:

  * Success (usually coded as `1`)
  * Failure (usually coded as `0`)

* The probability of success = $p$

* The probability of failure = $1-p$

---

# 🔹 Bernoulli Process

A **Bernoulli process** is a sequence of independent Bernoulli trials:

$$
X_1, X_2, X_3, \dots
$$

* Each $X_i \sim \text{Bernoulli}(p)$
* Independent and identically distributed (i.i.d).

Examples:

* Coin tosses (p = 0.5)
* Defective product probability in manufacturing
* Packet drop in networks

---

# 🔹 Distribution

* A single Bernoulli trial follows:

  $$
  P(X=1) = p, \quad P(X=0) = 1-p
  $$

* The sum of $n$ Bernoulli trials follows a **Binomial distribution**:

  $$
  S_n = X_1 + X_2 + \dots + X_n \sim \text{Binomial}(n,p)
  $$

---

# 🔹 Python Simulation

```python
import random

def bernoulli_trial(p=0.5):
    """Return 1 (success) with prob p, else 0 (failure)."""
    return 1 if random.random() < p else 0

def bernoulli_process(p=0.5, n=10):
    """Simulate n Bernoulli trials."""
    return [bernoulli_trial(p) for _ in range(n)]

# Example: 10 coin tosses
print(bernoulli_process(p=0.5, n=10))

# Example: biased coin (p=0.7 for heads)
print(bernoulli_process(p=0.7, n=10))
```

---

# 🔹 Example Output

```
[1, 0, 1, 1, 0, 0, 1, 1, 0, 1]   # fair coin
[1, 1, 1, 0, 1, 1, 0, 1, 1, 1]   # biased coin (p=0.7)
```

---

# 🔹 Applications

1. **Modeling randomness**

   * Tossing coins, yes/no events
2. **Machine learning**

   * Logistic regression is built on Bernoulli distribution
3. **Networking**

   * Modeling packet success/failure
4. **Reliability**

   * Probability of a system working or failing

---

✅ So:

* **Bernoulli trial** = one yes/no random event
* **Bernoulli process** = repeated independent trials
* **Binomial distribution** = total number of successes in n trials

---

Would you like me to also explain the **difference between Bernoulli process and Binomial distribution** (they’re closely related but used differently in problems)?
