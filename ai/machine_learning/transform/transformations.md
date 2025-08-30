This is a broad and deep topic, because **transformations** show up in math, stats, ML, and algorithms. Let’s organize it so you get a **map of the different types**.

---

# 🔹 What is a Transformation?

A **transformation** is simply a function that maps data/objects from one space to another, often to simplify analysis, computation, or representation.

Formally:

$$
T : X \to Y, \quad y = T(x)
$$

Depending on properties of $T$, we classify transformations differently.

---

# 🔹 1. **Linear Transformations**

* Definition: preserves **addition** and **scalar multiplication**:

$$
T(ax + by) = aT(x) + bT(y)
$$

* Examples:

  * Matrix multiplication ($y = Ax$)
  * Scaling, rotation, reflection, projection
* Used in: linear algebra, graphics, PCA, neural nets (linear layers).

---

# 🔹 2. **Affine Transformations**

* Linear + translation:

$$
T(x) = Ax + b
$$

* Preserves straight lines but not the origin.
* Examples: scaling, rotation, translation in computer graphics.

---

# 🔹 3. **Monotonic Transformations**

* Function that preserves order:

$$
x_1 < x_2 \implies f(x_1) < f(x_2) \quad (\text{increasing})
$$

or

$$
x_1 < x_2 \implies f(x_1) > f(x_2) \quad (\text{decreasing})
$$

* Examples:

  * Log, exponential, square root
  * Normalization in stats (rank-preserving)
* Used in: statistics (e.g., monotonic link functions), economics (utility functions).

---

# 🔹 4. **Nonlinear Transformations**

* Anything not linear.
* Examples: polynomial ($x^2$), logarithmic, sigmoid, tanh.
* Used in: machine learning (activations), feature engineering.

---

# 🔹 5. **Orthogonal / Isometric Transformations**

* Preserve distances and angles.
* Examples:

  * Rotation matrices
  * Reflections
* Used in: computer graphics, preserving geometric structure.

---

# 🔹 6. **Fourier / Spectral Transformations**

* Convert data from one domain → frequency domain.
* Examples:

  * Fourier Transform
  * Laplace Transform
  * Wavelet Transform
* Used in: signal processing, physics, ML feature extraction.

---

# 🔹 7. **Logarithmic / Power Transformations (Statistical)**

* Reduce skewness, stabilize variance.
* Examples:

  * $y = \log(x)$
  * $y = \sqrt{x}$
  * Box–Cox transform
* Used in: regression, econometrics.

---

# 🔹 8. **Probability & Distribution Transformations**

* Change random variables while preserving probability mass.
* Examples:

  * CDF transforms (Inverse Transform Sampling)
  * Normalization: $z = \frac{x-\mu}{\sigma}$
  * Copula transformations in multivariate stats

---

# 🔹 9. **Monotonic Stack / Order-Preserving Transforms**

* In algorithms, monotonic transformations are applied to maintain ordered structures (like min/max stacks).

---

# 🔹 10. **Nonlinear Geometric Transformations**

* Warp coordinates (not preserving linearity).
* Examples:

  * Perspective projection
  * Distortions in images
  * Nonlinear dimensionality reduction (t-SNE, UMAP)

---

# 🔹 11. **Group & Symmetry Transformations**

* From abstract algebra: transformation that preserves group structure.
* Examples:

  * Rotations (SO(3))
  * Lorentz transformations in relativity

---

# ✅ Summary Table

| Transformation Type     | Preserves                        | Example                    |
| ----------------------- | -------------------------------- | -------------------------- |
| Linear                  | Linearity (scaling + additivity) | Matrix multiplication      |
| Affine                  | Parallelism (not origin)         | Rotation + translation     |
| Monotonic               | Ordering                         | Log, exp                   |
| Nonlinear               | Nothing special                  | Sigmoid, tanh              |
| Orthogonal/Isometric    | Distances, angles                | Rotation                   |
| Fourier/Spectral        | Converts domains                 | Fourier Transform          |
| Statistical (Power/Log) | Stabilize variance               | Box-Cox                    |
| Probability             | Distribution                     | Inverse transform sampling |
| Geometric (Nonlinear)   | Shapes but distorted             | Perspective projection     |
| Group/Symmetry          | Algebraic structure              | Rotation groups            |

---

👉 Would you like me to also **connect this to neural networks**, i.e. how different transformations (linear, nonlinear, monotonic) are combined in layers to make models universal approximators?
