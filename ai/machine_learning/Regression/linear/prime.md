Yes 👍
Here’s the **same example using Linear Regression**, trained on the **first 20 prime numbers**, and then printing the **learned model equation**.

---

## 1️⃣ Create Dataset

We’ll predict a **continuous target**.
Example target:
👉 **y = 2x + noise** (simple and interpretable)

```python
import numpy as np
from sklearn.linear_model import LinearRegression

# First 20 prime numbers
X = np.array([
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
    31, 37, 41, 43, 47, 53, 59, 61, 67, 71
]).reshape(-1, 1)

# Target variable (linear relation)
y = 2 * X.flatten()
```

---

## 2️⃣ Train Linear Regression Model

```python
model = LinearRegression()
model.fit(X, y)
```

---

## 3️⃣ Print the Model Output Formula

Linear Regression formula:

[
y = wx + b
]

Extract parameters:

```python
w = model.coef_[0]
b = model.intercept_

print("Model Formula:")
print(f"y = {w:.2f} * x + {b:.2f}")
```

---

## 4️⃣ Example Output

```
y = 2.00 * x + 0.00
```

---

## 5️⃣ Interpretation

* **x** → prime number
* **w (slope)** → how much `y` increases per unit of `x`
* **b (intercept)** → baseline value when `x = 0`

---

## One-line takeaway (interview-ready)

> **Linear Regression learns a straight-line relationship of the form y = wx + b to predict continuous values.**

If you want next:

* Same example with **noise**
* **PyTorch implementation**
* Compare **Linear vs Logistic Regression visually**


Good question — this is a **direct consequence of how we constructed the data**.

---

## Why the bias (intercept) is 0

You defined the target as:

[
y = 2x
]

There is **no constant offset term** in the true data-generating process.

---

## What linear regression learns

Linear Regression learns:

[
y = wx + b
]

To minimize error:

* Best **slope** → ( w = 2 )
* Best **intercept** → ( b = 0 )

Because:

* When ( x = 0 ), the true ( y = 0 )
* Adding any non-zero bias would **increase error**

---

## Intuition

> The model learns exactly the function you gave it. Since the line passes through the origin, the intercept must be zero.

---

## If you add noise or offset

Example:

```python
y = 2 * X.flatten() + 5
```

Now the learned model becomes:

```
y ≈ 2x + 5
```

---

## Interview one-liner

> **The bias is zero because the underlying data has no constant offset; the optimal regression line passes through the origin.**

If you want, I can:

* Show the effect of noise
* Visualize intercept changes
* Explain bias in neural networks vs regression
