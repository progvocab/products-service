

The term **Propensity** generally means **a natural tendency, inclination, or likelihood** toward something.

---

### 🔹 In General English

* **Propensity** = a strong tendency or inclination to behave in a certain way.
  Example: *He has a propensity for taking risks.*

---

### 🔹 In Statistics & Machine Learning

In data science, **propensity** usually refers to a **probability or likelihood of an event occurring**, given certain conditions.

1. **Propensity Score**

   * Used in causal inference (e.g., A/B testing, treatment vs. control groups).
   * It’s the probability of receiving a treatment, given observed features.
   * Example: $P(Treatment = 1 \mid X)$.

2. **Propensity Modeling**

   * Building a predictive model to estimate the likelihood of a user taking an action.
   * Example: **Propensity to Buy** – predicting if a customer is likely to purchase.
   * Example: **Propensity to Churn** – predicting if a customer will leave a service.

3. **Marketing Use Cases**

   * Propensity models are used to segment users based on likelihood to perform actions:

     * Click an ad
     * Buy a product
     * Renew a subscription

---

### 🔹 Formula Example

If $X$ = customer features (age, income, past purchases),
and $Y$ = whether they purchase (1 = Yes, 0 = No),

A **propensity model** estimates:

$$
P(Y=1 \mid X) = \text{propensity to purchase}
$$

---

👉 Do you want me to show you how to **build a simple Propensity Model in Python** (say, logistic regression for predicting customer purchase)?
