**Hugging Face PEFT (Parameter-Efficient Fine-Tuning)** is a library that lets you **adapt large pretrained models using very few trainable parameters**.

* Instead of updating all model weights, PEFT trains **small add-on components** (like LoRA, adapters, prefix tuning).
* The base model stays **frozen**, saving memory and compute.
* Enables fast fine-tuning on **limited data or hardware**.
* Multiple task-specific adapters can be swapped on the same model.

**In short:** PEFT makes large models **cheap, fast, and practical to fine-tune**.
