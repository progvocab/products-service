Got it 👍 — let’s go step by step through the **core terminology in neural networks**, from the building blocks to how they learn.

---

# 🔹 Neural Network Terminology

## 1. **Neuron (Node, Unit)**

* Basic building block, inspired by a biological neuron.
* Takes inputs, applies weights + bias, passes through an **activation function**, outputs a value.

Formula:

$$
z = \sum (w_i \cdot x_i) + b
$$

$$
y = f(z)
$$

---

## 2. **Weights (w)**

* Parameters that determine the **importance of each input**.
* Learned during training by minimizing error.

---

## 3. **Bias (b)**

* Extra parameter that shifts the activation function curve.
* Helps model learn patterns that don’t pass through the origin.

---

## 4. **Activation Function (f)**

* Introduces **non-linearity** (otherwise NN = linear regression).
* Common types:

  * **Sigmoid** → outputs (0,1) for probabilities.
  * **ReLU** (Rectified Linear Unit) → fast, avoids vanishing gradient.
  * **Tanh** → outputs (-1,1).
  * **Softmax** → converts outputs into probabilities for classification.

---

## 5. **Layers**

* **Input Layer** → receives raw features.
* **Hidden Layers** → extract patterns via weights/activations.
* **Output Layer** → produces final prediction (class, value, etc.).

👉 A **Deep Neural Network (DNN)** = many hidden layers.

---

## 6. **Forward Propagation**

* Process of passing inputs → neurons → activations → final output.
* Prediction mode.

---

## 7. **Loss Function (Cost Function)**

* Measures **how wrong** the prediction is compared to true value.
* Examples:

  * MSE (Mean Squared Error) → regression.
  * Cross-Entropy → classification.

---

## 8. **Backpropagation**

* Algorithm for **updating weights**.
* Uses **chain rule of calculus** to compute gradients of loss w\.r.t weights.

---

## 9. **Gradient Descent**

* Optimization algorithm to minimize loss.
* Updates weights:

$$
w = w - \eta \cdot \frac{\partial L}{\partial w}
$$

where **η (eta)** = learning rate.

Variants: SGD, Adam, RMSProp.

---

## 10. **Epoch**

* One full pass through the training dataset.

---

## 11. **Batch Size**

* Number of training samples processed before weights are updated.
* **Mini-Batch Gradient Descent** is most common.

---

## 12. **Overfitting / Underfitting**

* **Overfitting** → model memorizes training data, performs poorly on new data.
* **Underfitting** → model too simple, fails to capture patterns.
* Fixes: dropout, regularization, more data.

---

## 13. **Dropout**

* Regularization technique: randomly “drop” (ignore) neurons during training to prevent overfitting.

---

## 14. **Epoch vs Iteration**

* **Epoch** = full dataset pass.
* **Iteration** = one batch update.

---

## 15. **Learning Rate**

* Step size for updating weights.
* Too high → unstable training.
* Too low → very slow learning.

---

## 16. **Hyperparameters**

* Settings chosen **before training** (not learned):

  * Learning rate
  * Batch size
  * Number of layers, neurons
  * Activation function

---

## 17. **Parameters**

* Learned values during training:

  * Weights
  * Biases

---

## 18. **Vanishing / Exploding Gradients**

* Common issue in deep networks.
* **Vanishing** → gradients shrink, network stops learning.
* **Exploding** → gradients grow too large, unstable training.
* Solutions: ReLU, normalization, gradient clipping.

---

# 🔹 Visual Summary

**Neural Network Flow**:
Input → Weights + Bias → Activation → Hidden Layers → Output → Loss → Backpropagation → Updated Weights

---

✅ With this foundation, you can now interpret **any deep learning paper or framework**.

Do you want me to also create a **diagram showing the neural network flow with these terms labeled** (like Input, Weights, Activation, Loss, Backprop)?
