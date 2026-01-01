**Torch (PyTorch)** is an **open-source deep learning framework** used to **build, train, and deploy neural networks**, especially for **deep learning and AI research/production**.

In short:

> **Torch = tensors + automatic differentiation + GPU acceleration**

---

## 1. What Problem Torch Solves

Before PyTorch:

* Static computation graphs (hard to debug)
* Limited flexibility for research

Torch introduced:

* **Dynamic computation graphs**
* Python-first API
* Easy debugging (like normal Python code)

---

## 2. Core Building Blocks

### 2.1 Tensors

* Multi-dimensional arrays (like NumPy)
* Can run on **CPU or GPU (CUDA)**

```python
import torch

x = torch.tensor([[1., 2.], [3., 4.]], device="cpu")
```

---

### 2.2 Autograd (Automatic Differentiation)

* Automatically computes gradients for backpropagation

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
y.backward()
print(x.grad)  # 4.0
```

---

### 2.3 Neural Network Module (`torch.nn`)

* Prebuilt layers and loss functions
* Enables modular model design

```python
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)
```

---

## 3. Training Loop (Core Concept)

PyTorch gives **full control** over training:

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

for x, y in data:
    optimizer.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    optimizer.step()
```

---

## 4. GPU & Performance

* Seamless CPU ↔ GPU transfer
* CUDA and distributed training support

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
```

---

## 5. Torch Ecosystem

| Library             | Purpose                |
| ------------------- | ---------------------- |
| `torchvision`       | Computer vision        |
| `torchaudio`        | Audio processing       |
| `torchtext`         | NLP                    |
| `torch.distributed` | Multi-GPU / multi-node |
| `torch.compile`     | Graph optimization     |

---

## 6. Torch vs TensorFlow (Quick)

| Aspect     | PyTorch   | TensorFlow      |
| ---------- | --------- | --------------- |
| Graph      | Dynamic   | Static (mostly) |
| Debugging  | Easy      | Harder          |
| Research   | Preferred | Less common     |
| Production | Strong    | Strong          |

---

## 7. Why Torch Is Popular

* Intuitive, Pythonic API
* Used by:

  * Meta (FAIR)
  * OpenAI
  * Tesla
  * Hugging Face
* Backbone of **Transformers**, **Diffusion models**, **LLMs**

---

## 8. Typical Use Cases

* Image classification (CNNs)
* NLP (Transformers, BERT, GPT)
* Speech recognition
* Reinforcement learning
* Recommendation systems

---

## 9. Torch vs NumPy (Important)

| Feature       | NumPy | Torch |
| ------------- | ----- | ----- |
| GPU support   | No    | Yes   |
| Autograd      | No    | Yes   |
| Deep learning | No    | Yes   |

---

## 10. Mental Model

> **Torch is NumPy + gradients + GPUs + neural networks.**

---

### One-Line Summary

**Torch (PyTorch)** is a flexible, Python-first deep learning framework that enables dynamic neural network training with automatic differentiation and GPU acceleration.

If you want, I can:

* Explain **torch vs keras**
* Show **how Transformers use torch internally**
* Walk through **distributed training on GPUs**
* Map torch usage to **production ML pipelines**
