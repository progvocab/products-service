Got it ✅ — let’s build a **comprehensive guide** to similarity search methods.
I’ll cover **the most widely used similarity/distance measures**, give you the **formula**, **when to use**, and **Python examples** with `NumPy`.

---

# 🔹 Similarity & Distance Measures

---

## 1. **Euclidean Distance (L2 Norm)**

📌 Formula:

$$
d(x,y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}
$$

📌 Use cases:

* Face recognition (FaceNet, DeepFace).
* Image embeddings similarity.
* Clustering (k-means).

📌 Python:

```python
import numpy as np

x = np.array([1, 2, 3])
y = np.array([2, 4, 6])

euclidean = np.linalg.norm(x - y)
print("Euclidean Distance:", euclidean)
```

---

## 2. **Manhattan Distance (L1 Norm)**

📌 Formula:

$$
d(x,y) = \sum_{i=1}^n |x_i - y_i|
$$

📌 Use cases:

* High-dimensional sparse data (bag-of-words).
* K-medians clustering.
* Robust to outliers.

📌 Python:

```python
manhattan = np.sum(np.abs(x - y))
print("Manhattan Distance:", manhattan)
```

---

## 3. **Cosine Similarity**

📌 Formula:

$$
\text{cosine}(x,y) = \frac{x \cdot y}{\|x\|\|y\|}
$$

📌 Use cases:

* Text embeddings (Word2Vec, BERT, TF-IDF).
* Recommender systems.
* High-dimensional embeddings.

📌 Python:

```python
cosine_sim = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
print("Cosine Similarity:", cosine_sim)
```

---

## 4. **Dot Product Similarity**

📌 Formula:

$$
\text{similarity}(x,y) = x \cdot y = \sum_i x_i y_i
$$

📌 Use cases:

* Neural recommender systems.
* Matrix factorization.
* Large-scale retrieval (with normalized vectors).

📌 Python:

```python
dot_product = np.dot(x, y)
print("Dot Product:", dot_product)
```

---

## 5. **Jaccard Similarity** (Set-based)

📌 Formula:

$$
J(A,B) = \frac{|A \cap B|}{|A \cup B|}
$$

📌 Use cases:

* Text/documents (bag of words).
* Binary features (yes/no attributes).
* Recommender systems (shared items).

📌 Python:

```python
A = set([1,2,3,4])
B = set([3,4,5,6])
jaccard = len(A & B) / len(A | B)
print("Jaccard Similarity:", jaccard)
```

---

## 6. **Hamming Distance**

📌 Formula:

$$
d(x,y) = \sum_{i=1}^n [x_i \neq y_i]
$$

📌 Use cases:

* DNA sequences.
* Binary strings.
* Error detection in codes.

📌 Python:

```python
x = np.array([1,0,1,1])
y = np.array([1,1,0,1])
hamming = np.sum(x != y)
print("Hamming Distance:", hamming)
```

---

## 7. **Chebyshev Distance (L∞ norm)**

📌 Formula:

$$
d(x,y) = \max_i |x_i - y_i|
$$

📌 Use cases:

* Chessboard distance (king’s moves).
* Quality control (max deviation).

📌 Python:

```python
x = np.array([1, 2, 3])
y = np.array([2, 4, 6])
chebyshev = np.max(np.abs(x - y))
print("Chebyshev Distance:", chebyshev)
```

---

## 8. **Mahalanobis Distance**

📌 Formula:

$$
d(x,y) = \sqrt{(x-y)^T S^{-1} (x-y)}
$$

where $S$ = covariance matrix.

📌 Use cases:

* Accounts for feature correlation.
* Anomaly detection.
* Multivariate outlier detection.

📌 Python:

```python
from scipy.spatial import distance

data = np.array([[1,2,3],[2,4,6],[3,6,9]])
cov = np.cov(data.T)
inv_cov = np.linalg.inv(cov)

x, y = data[0], data[1]
mahalanobis = distance.mahalanobis(x, y, inv_cov)
print("Mahalanobis Distance:", mahalanobis)
```

---

## 9. **Dynamic Time Warping (DTW)**

📌 Formula: Aligns two time-series by warping the time axis.

📌 Use cases:

* Speech recognition.
* Gesture recognition.
* Time-series similarity.

📌 Python:

```python
from dtw import dtw

x = np.array([1,2,3,4,2])
y = np.array([1,1,2,3,4])

dist, cost, acc_cost, path = dtw(x.reshape(-1,1), y.reshape(-1,1))
print("DTW Distance:", dist)
```

---

## 10. **Pearson Correlation Similarity**

📌 Formula:

$$
\text{similarity}(x,y) = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2}\sqrt{\sum (y_i - \bar{y})^2}}
$$

📌 Use cases:

* Collaborative filtering.
* Measuring linear correlation.

📌 Python:

```python
pearson = np.corrcoef(x, y)[0,1]
print("Pearson Correlation:", pearson)
```

---

# ✅ Summary Table

(blank line before table for your preference)

| Method      | Formula                        | Range              | Best for             | Notes                |                      |                    |                   |               |
| ----------- | ------------------------------ | ------------------ | -------------------- | -------------------- | -------------------- | ------------------ | ----------------- | ------------- |
| Euclidean   | $\sqrt{\sum (x_i-y_i)^2}$      | $[0,\infty)$       | Geometric similarity | Sensitive to scale   |                      |                    |                   |               |
| Manhattan   | (\sum                          | x\_i-y\_i          | )                    | $[0,\infty)$         | Sparse/high-dim data | Robust to outliers |                   |               |
| Cosine      | $\frac{x \cdot y}{\|x\|\|y\|}$ | $[-1,1]$           | Embeddings, text     | Ignores magnitude    |                      |                    |                   |               |
| Dot Product | $\sum x_i y_i$                 | $(-\infty,\infty)$ | Recommenders         | Needs normalization  |                      |                    |                   |               |
| Jaccard     | (\frac{                        | A\cap B            | }{                   | A\cup B              | })                   | $[0,1]$            | Sets, binary data | Good for text |
| Hamming     | $\sum [x_i \neq y_i]$          | $[0,n]$            | DNA, binary codes    | Counts mismatches    |                      |                    |                   |               |
| Chebyshev   | (\max                          | x\_i-y\_i          | )                    | $[0,\infty)$         | Chess moves, QC      | Max deviation      |                   |               |
| Mahalanobis | $\sqrt{(x-y)^T S^{-1}(x-y)}$   | $[0,\infty)$       | Correlated features  | Needs covariance     |                      |                    |                   |               |
| DTW         | Warped path distance           | $[0,\infty)$       | Time series          | Non-linear alignment |                      |                    |                   |               |
| Pearson     | Correlation coeff.             | $[-1,1]$           | Ratings, correlation | Captures linearity   |                      |                    |                   |               |

---

👉 Do you want me to extend this into a **ready-to-run notebook** (with a dataset and k-NN retrieval using each metric), so you can see how search results differ depending on similarity?
