In Transformers, an **adapter** is a **small, trainable module inserted inside a pretrained model** to adapt it to a new task or domain.

* It sits between existing Transformer layers (often after attention or FFN).
* The original model weights are **frozen**; only the adapter is trained.
* Adapters are **lightweight** (few parameters), making fine-tuning efficient.
* Multiple adapters can be swapped for different tasks without retraining the full model.

**In short:** adapters let you **customize large Transformers cheaply and safely**.
