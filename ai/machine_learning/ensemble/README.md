### One-line explanations of ensemble strategies

* **Ensemble (definition)**: Combining predictions from multiple models to improve accuracy, robustness, and generalization.

* **Bagging (Bootstrap Aggregating)**: Train models independently on bootstrapped samples and average/vote to reduce variance.

* **Random Forest**: Bagging of decision trees with random feature selection to further decorrelate models and reduce overfitting.

* **Boosting**: Train models sequentially, each focusing more on previous errors to reduce bias.

* **AdaBoost**: A boosting method that reweights misclassified samples so later models focus on hard cases.

* **Gradient Boosting**: Builds models sequentially by fitting each one to the residual errors of the previous ensemble.

* **XGBoost**: Optimized gradient boosting with regularization, parallelism, and efficient handling of sparse data.

* **LightGBM**: Histogram-based gradient boosting that grows trees leaf-wise for high speed and large datasets.

* **CatBoost**: Gradient boosting optimized for categorical features with minimal preprocessing.

* **Stacking**: Train multiple diverse base models and a meta-model that learns how to combine their predictions.

* **Blending**: A simplified stacking approach using a hold-out validation set instead of cross-validation.

* **Voting (Hard/Soft)**: Combine model predictions by majority vote (hard) or averaged probabilities (soft).

* **Averaging**: Combine predictions by taking the mean (common for regression).

* **Weighted Ensemble**: Combine models using weights based on their performance.

* **Isolation Forest**: An ensemble of random trees designed to isolate anomalies quickly.

---

### Interview-ready one-liner

> **Ensemble learning improves model performance by combining multiple diverse models so that their errors cancel out and predictions become more stable and accurate.**
