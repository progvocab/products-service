The **bias–variance decomposition** of model error is expressed as:

[
\mathbb{E}[(y - \hat{f}(x))^2]
==============================

\underbrace{\text{Bias}^2}*{\text{error from wrong assumptions}}
+
\underbrace{\text{Variance}}*{\text{error from data sensitivity}}
+
\underbrace{\sigma^2}_{\text{irreducible noise}}
]

### Meaning in simple terms

* **Bias²**: Error because the model is too simple
* **Variance**: Error because the model is too complex
* **Noise (σ²)**: Error that cannot be reduced

This formula explains why increasing complexity helps until variance dominates.


* **Bias** is error from overly simple assumptions in the model.
* High bias causes **underfitting** (model misses patterns).
* **Variance** is error from sensitivity to training data.
* High variance causes **overfitting** (model memorizes noise).
* Increasing model complexity **reduces bias but increases variance**.
* Simplifying a model **reduces variance but increases bias**.
* Data augmentation and regularization reduce **variance**.
* More data helps lower **variance** without raising bias.
* Proper model selection balances both errors.
* The goal is **minimum total error**, not zero bias or variance.


In addition to **bias and variance (error components)**, a machine learning model has several **core components**, each playing a specific role:

* **Parameters** – Learnable values (weights and biases) that define the model’s behavior.
* **Hyperparameters** – User-defined settings (learning rate, layers, units, epochs).
* **Architecture** – Structure of the model (layers, connections, depth).
* **Loss Function** – Measures how wrong the model’s predictions are.
* **Optimizer** – Updates parameters to minimize the loss.
* **Metrics** – Evaluate performance (accuracy, precision, recall).
* **Training Data** – Examples the model learns from.
* **Validation/Test Data** – Used to evaluate generalization.
* **Regularization** – Techniques to control overfitting (dropout, L2).
* **Inference Logic** – How predictions are generated after training.

Together, these define **how the model learns, evaluates, and predicts**.
