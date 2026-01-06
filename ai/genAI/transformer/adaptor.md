In Transformers, an **adapter** is a **small, trainable module inserted inside a pretrained model** to adapt it to a new task or domain.

* It sits between existing Transformer layers (often after attention or FFN).
* The original model weights are **frozen**; only the adapter is trained.
* Adapters are **lightweight** (few parameters), making fine-tuning efficient.
* Multiple adapters can be swapped for different tasks without retraining the full model.

**In short:** adapters let you **customize large Transformers cheaply and safely**.


Here is a **very simple and minimal example** showing a **Transformer block with an Adapter**, using **TensorFlow / Keras**.

### Simple Transformer + Adapter (conceptual example)

```python
import tensorflow as tf
from tensorflow.keras import layers

# Simple Adapter layer
class Adapter(layers.Layer):
    def __init__(self, hidden_dim):
        super().__init__()
        self.down = layers.Dense(hidden_dim // 4, activation="relu")
        self.up = layers.Dense(hidden_dim)

    def call(self, x):
        return x + self.up(self.down(x))  # residual connection


# Simple Transformer block
class SimpleTransformer(layers.Layer):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = layers.MultiHeadAttention(num_heads=2, key_dim=hidden_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(hidden_dim * 2, activation="relu"),
            layers.Dense(hidden_dim)
        ])
        self.adapter = Adapter(hidden_dim)
        self.norm = layers.LayerNormalization()

    def call(self, x):
        x = self.norm(x + self.attn(x, x))
        x = self.norm(x + self.ffn(x))
        x = self.adapter(x)   # adapter added here
        return x
```

### What this shows (in simple terms)

* Transformer learns **general features** (attention + FFN).
* Adapter is a **small plug-in** trained for a specific task.
* Base model can stay **frozen**, adapter learns fast.

This is exactly how adapters are used in **real-world fine-tuning of large Transformers**.
