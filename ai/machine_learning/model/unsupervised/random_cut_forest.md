 **Random Cut Forest (RCF)** is a fascinating algorithm, especially in **unsupervised anomaly detection**. Let’s go step by step.

---

## 🔹 What is Random Cut Forest?

* **RCF** = an **unsupervised learning algorithm** for **anomaly detection**.
* Developed at Amazon (used in Amazon SageMaker).
* Works well on **high-dimensional, streaming data**.
* It detects anomalies by modeling data distribution with an ensemble of **random cut trees**.

Think of it as a forest of decision trees built using **random cuts** of the data space. The algorithm measures how “surprising” a point is compared to the rest → higher surprise = anomaly.

---

## 🔹 Core Idea

1. Build a **Random Cut Tree (RCTree)**:

   * Start with all data points in a bounding box.
   * Randomly pick a dimension and cut position to split points into two groups.
   * Recurse until each point is isolated.
   * Each point ends up as a leaf.

2. **Path length** = the number of cuts needed to isolate a point.

   * Normal points (dense regions) → take **longer** to isolate.
   * Anomalies (sparse/outliers) → take **shorter** to isolate.

3. Use an **ensemble of trees (forest)** to reduce randomness.

   * Compute an **anomaly score** = average normalized path length.

---

## 🔹 Formula (Anomaly Score)

Let $h(x)$ = average path length of point $x$ across trees.
Normalized anomaly score:

$$
s(x) = 2^{-\frac{h(x)}{c(n)}}
$$

* $c(n)$ = average path length for $n$ points in a tree (normalization factor).
* Higher $s(x)$ = more anomalous.

---

## 🔹 Advantages

* ✅ Works in high-dimensional spaces.
* ✅ Online-friendly → can handle **streaming data** (insert/delete points efficiently).
* ✅ No strong assumptions about data distribution.
* ✅ Scalable (forest ensemble approach).

---

## 🔹 Use Cases

* Fraud detection (credit card, transactions).
* Intrusion detection (network traffic anomalies).
* Sensor fault detection (IoT, manufacturing).
* Log analysis (system errors, rare events).
* Time-series anomaly detection (RCF can be adapted).

---

## 🔹 Simple Python Example

```python
import numpy as np
from rrcf import RCTree

# Generate sample data (normal cluster + anomalies)
np.random.seed(0)
normal = np.random.randn(100, 2) * 0.5
anomalies = np.random.uniform(low=-5, high=5, size=(5, 2))
X = np.vstack([normal, anomalies])

# Build RCF forest
num_trees = 40
tree_size = 256
forest = []
for _ in range(num_trees):
    # Sample points with replacement
    ixs = np.random.choice(len(X), size=tree_size, replace=False)
    tree = RCTree(X[ixs])
    forest.append(tree)

# Compute anomaly scores
avg_codisp = {}
for i in range(len(X)):
    score = 0
    for tree in forest:
        if i in tree.leaves:
            score += tree.codisp(i)
    avg_codisp[i] = score / num_trees

# Show top anomalies
anomalies_detected = sorted(avg_codisp.items(), key=lambda x: x[1], reverse=True)[:5]
print("Top anomalies:", anomalies_detected)
```

Here:

* **codisp** = Co-Dispersion score (how isolated the point is).
* Higher score = more anomalous.

---

## 🔹 Comparison with Other Methods

(blank line before table for your preference)

| Algorithm             | Core Idea                            | Best for                         | Weakness                         |
| --------------------- | ------------------------------------ | -------------------------------- | -------------------------------- |
| **Isolation Forest**  | Random splits isolate points quickly | General anomaly detection        | Can struggle with streaming      |
| **One-Class SVM**     | Boundary around normal points        | Small/medium data                | Expensive in high-dim            |
| **Random Cut Forest** | Random cuts + path length            | High-dim, streaming, time series | Less known than Isolation Forest |

---

✅ **Summary:**
Random Cut Forest is an **unsupervised anomaly detection** algorithm that uses random partitions of data to isolate points. Points that are isolated quickly (short paths) are flagged as anomalies. It’s especially useful for **streaming, high-dimensional, and real-world anomaly detection tasks**.

---

Would you like me to also explain **how RCF extends to time-series anomaly detection** (where it detects unusual spikes, drops, or shifts)?
