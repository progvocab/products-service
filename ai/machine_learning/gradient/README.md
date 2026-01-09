Here is a **clean, interview-ready list of gradient (descent) variants**, each with a **one-line explanation**:

---

### Core Gradient Descent Types

1. **Batch Gradient Descent** – Computes gradient using the **entire dataset** per update (stable but slow).
2. **Stochastic Gradient Descent (SGD)** – Updates weights using **one data point at a time** (fast but noisy).
3. **Mini-Batch Gradient Descent** – Uses **small batches** of data (best balance, most common).

---

### Momentum-Based Methods

4. **Momentum** – Accelerates SGD by adding **velocity from past gradients**.
5. **Nesterov Accelerated Gradient (NAG)** – Looks ahead before computing gradient for **better convergence**.

---

### Adaptive Learning Rate Methods

6. **Adagrad** – Adapts learning rate per parameter, **good for sparse data**.
7. **Adadelta** – Fixes Adagrad’s shrinking learning rate problem.
8. **RMSProp** – Uses moving average of squared gradients for **stable updates**.

---

### Popular Optimizers (Industry Standard)

9. **Adam** – Combines **Momentum + RMSProp**, fast and widely used.
10. **AdamW** – Adam with **decoupled weight decay** (better generalization).
11. **Nadam** – Adam + **Nesterov momentum**.
12. **AMSGrad** – Adam variant with **guaranteed convergence**.

---

### Second-Order / Advanced

13. **Newton’s Method** – Uses **second derivatives (Hessian)** for fast convergence (expensive).
14. **L-BFGS** – Memory-efficient approximation of Newton’s method.
15. **Natural Gradient Descent** – Uses **information geometry** for better optimization.

---

### One-line interview summary

> **Gradient descent variants differ in how much data they use, how they adapt learning rates, and how they accelerate convergence.**

If you want, I can:

* Recommend **best optimizer per problem**
* Explain **why Adam is default**
* Show **PyTorch optimizer code comparison**
