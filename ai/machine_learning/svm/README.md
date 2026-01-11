### SVM — Simple explanation

**Support Vector Machine (SVM)** is a supervised ML algorithm that **separates data by drawing the best possible boundary (line or plane)** between classes, such that the **margin between classes is maximum**.

Think of it as:
👉 *“Find the widest gap that separates the data.”*

---

### Most common use cases of SVM

1. **Text classification**

   * Spam vs non-spam
   * Sentiment analysis
   * Document categorization
     *(Very popular because SVM works well in high-dimensional text data)*

2. **Binary classification problems**

   * Fraud vs normal transaction
   * Defective vs non-defective product
   * Disease present vs absent

3. **Image classification (small datasets)**

   * Face vs non-face
   * Object presence detection

4. **Anomaly / outlier detection (One-Class SVM)**

   * Network intrusion detection
   * Equipment fault detection (e.g., wind turbine anomalies)

5. **Regression (SVR)**

   * Price prediction
   * Load or demand forecasting

---

### Why SVM is commonly used

* Works well with **small to medium datasets**
* Effective in **high-dimensional spaces**
* Can model **non-linear boundaries** using kernels (RBF, polynomial)

---

### One-line interview answer

> **SVM is a supervised algorithm that finds the optimal boundary with maximum margin, commonly used for text classification, binary classification, and anomaly detection.**

If you want, I can also explain:

* Kernel trick in 2 lines
* SVM vs Logistic Regression
* SVM example in Python
* Why SVM works well for text data
