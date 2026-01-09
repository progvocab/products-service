Here is a **minimal, standard PyTorch GAN code snippet** to **generate synthetic data** (simple 2D data example). This is the **cleanest learning version**.

---

## Simple GAN for Data Generation (PyTorch)

### Generator & Discriminator

```python
import torch
import torch.nn as nn

# Generator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, z):
        return self.net(z)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
```

---

### Training Step (Core GAN Logic)

```python
G = Generator()
D = Discriminator()

loss_fn = nn.BCELoss()
g_opt = torch.optim.Adam(G.parameters(), lr=0.001)
d_opt = torch.optim.Adam(D.parameters(), lr=0.001)

# Real data (dummy example)
real_data = torch.randn(64, 2)

# -------- Train Discriminator --------
z = torch.randn(64, 10)
fake_data = G(z)

real_labels = torch.ones(64, 1)
fake_labels = torch.zeros(64, 1)

d_loss = loss_fn(D(real_data), real_labels) + \
         loss_fn(D(fake_data.detach()), fake_labels)

d_opt.zero_grad()
d_loss.backward()
d_opt.step()

# -------- Train Generator --------
z = torch.randn(64, 10)
fake_data = G(z)

g_loss = loss_fn(D(fake_data), real_labels)

g_opt.zero_grad()
g_loss.backward()
g_opt.step()
```

---

### Generate New Synthetic Data

```python
z = torch.randn(100, 10)
synthetic_data = G(z)
print(synthetic_data[:5])
```

---

## What this code does

* Learns an **implicit data distribution**
* Generates **new synthetic samples**
* No labels required (unsupervised)

---

## One-line takeaway

> **GAN generates data by training a generator to fool a discriminator into believing synthetic samples are real.**

If you want next:

* **Image GAN (MNIST)**
* **Conditional GAN**
* **Fraud data GAN**
* **GAN vs Diffusion code**
