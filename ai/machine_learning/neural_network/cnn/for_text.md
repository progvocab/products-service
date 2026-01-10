Here’s a **simple, classic CNN-for-text example** (sentence classification) — short, clear, and interview-friendly.

---

## Example: CNN for Text Classification (PyTorch)

### Task

Classify a sentence as **positive or negative**

---

### Model

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.conv1 = nn.Conv1d(embed_dim, 100, kernel_size=3)
        self.conv2 = nn.Conv1d(embed_dim, 100, kernel_size=4)
        self.conv3 = nn.Conv1d(embed_dim, 100, kernel_size=5)

        self.fc = nn.Linear(300, 1)

    def forward(self, x):
        x = self.embedding(x)           # (batch, seq_len, embed_dim)
        x = x.permute(0, 2, 1)          # (batch, embed_dim, seq_len)

        c1 = F.relu(self.conv1(x))
        c2 = F.relu(self.conv2(x))
        c3 = F.relu(self.conv3(x))

        p1 = F.max_pool1d(c1, c1.size(2)).squeeze(2)
        p2 = F.max_pool1d(c2, c2.size(2)).squeeze(2)
        p3 = F.max_pool1d(c3, c3.size(2)).squeeze(2)

        out = torch.cat([p1, p2, p3], dim=1)
        return torch.sigmoid(self.fc(out))
```

---

## How convolution works on text

```
Sentence → Embeddings → 1D Convolution → n-gram features → Classification
```

* Kernel size 3 → trigrams
* Kernel size 4 → 4-grams
* Kernel size 5 → 5-grams

---

## Why CNNs work for text

* Capture **local word patterns**
* Fast & parallel
* Less sequential dependency than RNNs

---

## One-line interview answer

> **CNNs for text apply 1D convolutions over word embeddings to learn n-gram features for classification.**

If you want:

* CNN vs LSTM vs Transformer for text
* Minimal runnable example
* Real dataset demo
