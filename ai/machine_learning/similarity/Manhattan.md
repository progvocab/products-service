Nice one 👍 — let’s go through **Manhattan distance / similarity** step by step, and compare it with Euclidean and cosine.

---

## 🔹 Manhattan Distance

Also called **L1 distance** or **taxicab distance**:

$$
d_{\text{Manhattan}}(x, y) = \sum_{i=1}^{n} |x_i - y_i|
$$

* It measures the distance if you can only move **horizontally and vertically** (like a taxi driving in a city grid).
* Unlike Euclidean (straight line), Manhattan adds up absolute differences along each axis.

👉 Example in 2D:

* $x = (1, 2)$, $y = (3, 5)$
* Manhattan: $|1-3| + |2-5| = 2 + 3 = 5$
* Euclidean: $\sqrt{(1-3)^2 + (2-5)^2} = \sqrt{13} \approx 3.61$

---

## 🔹 Manhattan Similarity

Like Euclidean, Manhattan distance is a **dissimilarity measure**.
To turn it into similarity, we usually transform it:

1. **Inverse transformation**

   $$
   \text{similarity}(x,y) = \frac{1}{1 + d_{\text{Manhattan}}(x,y)}
   $$

2. **Exponential decay** (used in kernels/ML)

   $$
   \text{similarity}(x,y) = e^{-\alpha \, d_{\text{Manhattan}}(x,y)}
   $$

   where $\alpha$ controls sensitivity.

---

## 🔹 When Manhattan is Useful

* **High-dimensional sparse data**

  * In text mining, user ratings, or bag-of-words vectors, features are often sparse (mostly zeros).
  * Manhattan works better than Euclidean because it doesn’t overweight large coordinate differences.

* **Robust to outliers**

  * Euclidean squares differences (large deviations dominate).
  * Manhattan grows linearly, so it's less sensitive to outliers.

* **L1-regularization (Lasso)**

  * Encourages sparsity in ML models → closely related to Manhattan distance.

---

## 🔹 ML / Similarity Search Examples

1. **k-NN search**

   * Using Manhattan distance instead of Euclidean gives different neighborhoods (especially in high dimensions).

2. **Clustering (K-medians)**

   * If you replace L2 norm with L1 norm, centroids are computed using medians instead of means.
   * Works better with noisy or sparse data.

3. **Recommender systems**

   * User/item rating vectors can be compared via Manhattan similarity.
   * Example:

     * User A: \[5, 0, 3]
     * User B: \[4, 0, 1]
     * Manhattan distance = |5-4| + |0-0| + |3-1| = 1 + 0 + 2 = 3
     * Similarity = 1/(1+3) = 0.25

---

## 🔹 Comparison

| Metric             | Formula                        | Captures               | Pros                                  | Cons                         |                                                   |                             |
| ------------------ | ------------------------------ | ---------------------- | ------------------------------------- | ---------------------------- | ------------------------------------------------- | --------------------------- |
| **Euclidean (L2)** | $\sqrt{\sum (x_i - y_i)^2}$    | Straight-line distance | Smooth, good in low-dim spaces        | Sensitive to scale, outliers |                                                   |                             |
| **Manhattan (L1)** | (\sum                          | x\_i - y\_i            | )                                     | Grid-based path              | Robust to outliers, good for sparse/high-dim data | Less “natural” for geometry |
| **Cosine**         | $\frac{x \cdot y}{\|x\|\|y\|}$ | Angle between vectors  | Scale-invariant, works for embeddings | Ignores magnitude            |                                                   |                             |

---

✅ **Summary:**

* Manhattan distance = sum of absolute differences.
* Manhattan similarity is derived from it (e.g., inverse or exponential).
* Good for **sparse, high-dimensional, or noisy data**.
* Often used in ML clustering, recommender systems, and when robustness matters.

---

👉 Do you want me to **show you a Python example** comparing **Euclidean, Manhattan, and Cosine similarities** on the same dataset so you can see how neighbors change depending on the metric?
