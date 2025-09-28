A **DNN (Deep Neural Network)** is a type of **artificial neural network (ANN)** that has **multiple hidden layers** between the input and output layers.

It’s called "deep" because of the depth (number of layers). These extra layers allow the network to learn **complex, non-linear relationships** in the data.

---

## 🧩 Structure of a DNN

1. **Input Layer** – Takes raw data (features).
2. **Hidden Layers** – Multiple layers of neurons where computation happens.

   * Each neuron applies a **linear transformation** (weights × inputs + bias).
   * Then applies a **non-linear activation function** (ReLU, sigmoid, tanh, etc.).
3. **Output Layer** – Produces the final result (e.g., classification probabilities, recommendation scores).

---

### Example: Simple 3-Layer DNN

```
Input → Hidden Layer 1 → Hidden Layer 2 → Output
```

Mathematically:
[
h_1 = f(W_1 \cdot x + b_1)
]
[
h_2 = f(W_2 \cdot h_1 + b_2)
]
[
y = g(W_3 \cdot h_2 + b_3)
]

Where:

* (x) = input features
* (W, b) = weights and biases (learned during training)
* (f, g) = activation functions
* (y) = output prediction

---

## ✅ Why DNNs Are Powerful

* They can approximate **any continuous function** (Universal Approximation Theorem).
* Handle **non-linear patterns** that simple linear models can’t.
* Scalable to high-dimensional data (images, text, audio, user-item interactions).

---

## 📌 Example in PyTorch

```python
import torch
import torch.nn as nn

class SimpleDNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x)

# Example
model = SimpleDNN(input_dim=10, hidden_dim=64, output_dim=1)
sample_input = torch.rand(5, 10)  # batch of 5 samples
output = model(sample_input)
print(output)
```

---

## 🎯 Applications of DNNs

* **Recommendation Systems** (YouTube, Amazon) → Two-tower DNN models.
* **Computer Vision** (image classification, object detection).
* **NLP** (transformers are deep neural networks).
* **Fraud Detection, Forecasting, Healthcare AI**.

---

Would you like me to show you **how a DNN replaces traditional matrix factorization in recommendation systems** (like YouTube’s two-tower model)?
