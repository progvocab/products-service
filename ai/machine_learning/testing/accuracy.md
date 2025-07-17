Understanding metrics like **accuracy, precision, recall, F1-score, ROC-AUC**, etc., is critical for evaluating the performance of **classification models** in machine learning. Here's a complete explanation with examples:

---

## ✅ 1. **Accuracy**

**Definition:**
The proportion of total correct predictions.

**Formula:**

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

**Best used when:** Classes are balanced.

**Example:**
If your model made 90 correct predictions out of 100 total:

$$
\text{Accuracy} = \frac{90}{100} = 0.90 = 90\%
$$

---

## ✅ 2. **Precision**

**Definition:**
The proportion of positive predictions that were actually correct.

**Formula:**

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

**Best used when:** **False positives are costly.**

**Example:** In spam detection, precision ensures that actual emails are not wrongly classified as spam.

---

## ✅ 3. **Recall (Sensitivity / True Positive Rate)**

**Definition:**
The proportion of actual positives correctly identified.

**Formula:**

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

**Best used when:** **False negatives are costly.**

**Example:** In disease detection, missing a true case is dangerous, so recall is critical.

---

## ✅ 4. **F1-Score**

**Definition:**
The harmonic mean of precision and recall.

**Formula:**

$$
F1 = 2 \times \frac{Precision \cdot Recall}{Precision + Recall}
$$

**Best used when:** You want a **balance between precision and recall**, especially with imbalanced classes.

---

## ✅ 5. **Specificity (True Negative Rate)**

**Definition:**
The proportion of actual negatives correctly identified.

**Formula:**

$$
\text{Specificity} = \frac{TN}{TN + FP}
$$

**Used with:** Often along with recall to evaluate binary classifiers.

---

## ✅ 6. **ROC Curve (Receiver Operating Characteristic Curve)**

**Definition:**
A plot of **True Positive Rate (Recall)** vs **False Positive Rate** at various thresholds.

* **X-axis:** FPR = FP / (FP + TN)
* **Y-axis:** TPR = TP / (TP + FN)

---

## ✅ 7. **AUC (Area Under Curve)**

**Definition:**
The area under the ROC curve. Value between 0.5 (random) and 1 (perfect).

**Use case:** Compare models independent of threshold.

---

## 📊 Confusion Matrix

|                     | Predicted Positive  | Predicted Negative  |
| ------------------- | ------------------- | ------------------- |
| **Actual Positive** | TP (True Positive)  | FN (False Negative) |
| **Actual Negative** | FP (False Positive) | TN (True Negative)  |

---

## 🔍 Example

Suppose:

* TP = 70
* TN = 20
* FP = 5
* FN = 5

Then:

| Metric    | Value                                   |
| --------- | --------------------------------------- |
| Accuracy  | 90%                                     |
| Precision | 70 / (70+5) = 0.933                     |
| Recall    | 70 / (70+5) = 0.933                     |
| F1 Score  | 0.933                                   |
| AUC       | High (since TPR is high and FPR is low) |

---

## 🎯 Summary Table

| Metric    | Focuses On          | Best When...               |
| --------- | ------------------- | -------------------------- |
| Accuracy  | Overall correctness | Classes are balanced       |
| Precision | False Positives     | False positives are costly |
| Recall    | False Negatives     | False negatives are costly |
| F1-Score  | Balance             | Both FP and FN matter      |
| AUC       | Threshold-Free Eval | Comparing classifiers      |

---

Let me know if you'd like Python code to compute these metrics from predictions.
