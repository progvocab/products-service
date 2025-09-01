 let’s go step by step and break down the **main model artifacts** you’ll encounter in MLflow (and more generally in ML pipelines). Each format has different trade-offs in terms of **portability, security, performance, and ecosystem support**.

---

# 🔎 Common Model Artifacts

### 1. **Pickle (`.pkl` / `.sav`)**

* **What it is:**
  Python’s built-in object serialization format. Used by libraries like scikit-learn and XGBoost to save models.
* **How it works:**
  Serializes entire Python objects (including functions, classes, etc.) into a binary stream.
* **Pros:**

  * Very easy to use (`pickle.dump`, `pickle.load`).
  * Retains exact Python object (no conversion).
* **Cons / Risks:**

  * **Security risk**: Pickle is executable — loading untrusted pickle can run arbitrary code.
  * Not portable across Python versions or platforms.
  * Harder to integrate with non-Python environments.
* **Typical Use Cases:**

  * Saving intermediate models in a Python-only environment.
  * Quick prototyping in scikit-learn.

✅ Example:

```python
import pickle
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model.pkl", "rb") as f:
    loaded_model = pickle.load(f)
```

---

### 2. **Joblib (`.joblib`)**

*(Closely related to pickle, but optimized)*

* **What it is:**
  A wrapper around pickle with better handling of large NumPy arrays.
* **Pros:**

  * Faster for models with large numerical arrays.
  * Standard in scikit-learn.
* **Cons:**

  * Inherits pickle’s security risks.
* **Use Case:**
  Saving scikit-learn models with large arrays.

---

### 3. **ONNX (`.onnx`)**

* **What it is:**
  **Open Neural Network Exchange** format — an open standard for representing ML models across frameworks.
* **How it works:**
  Defines computation graph + parameters in a framework-neutral way.
* **Pros:**

  * **Portability:** Convert models from PyTorch, TensorFlow, scikit-learn → ONNX → run anywhere.
  * Supported by many runtimes: ONNX Runtime, TensorRT, OpenVINO.
  * Optimized inference (e.g., hardware accelerators).
* **Cons:**

  * Conversion sometimes lossy (not all layers supported).
  * Debugging can be harder.
* **Use Cases:**

  * Serving models in production across different stacks.
  * Deploying ML models to edge devices or mobile.

✅ Example (PyTorch → ONNX):

```python
import torch
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "model.onnx")
```

---

### 4. **TorchScript (`.pt`, `.pth`)**

* **What it is:**
  A serialized PyTorch model in TorchScript format.
* **How it works:**
  Converts dynamic PyTorch code into a static computation graph for efficient deployment.
* **Pros:**

  * Native to PyTorch ecosystem.
  * Can run independently of Python (e.g., C++ runtime).
  * Good for serving models in production (e.g., TorchServe).
* **Cons:**

  * Not portable outside PyTorch ecosystem (unlike ONNX).
  * Conversion from dynamic Python code to TorchScript may fail.
* **Use Cases:**

  * PyTorch production deployment.
  * Mobile inference (PyTorch Mobile).

✅ Example:

```python
scripted_model = torch.jit.script(model)
scripted_model.save("model.pt")
```

---

### 5. **TensorFlow SavedModel (`saved_model.pb` + variables folder)**

* **What it is:**
  TensorFlow’s standard format to save a complete model (graph, weights, signatures).
* **How it works:**
  Directory-based format:

  * `saved_model.pb` (protocol buffer defining computation graph).
  * `variables/` (trained weights).
* **Pros:**

  * Self-contained and versioned.
  * Works with TensorFlow Serving, TFLite, TensorFlow\.js.
  * Cross-platform (Python, C++, Java, Go).
* **Cons:**

  * TensorFlow-specific.
  * Can be large and complex to manage.
* **Use Cases:**

  * TensorFlow/Keras production deployment.
  * Converting to TFLite or TensorFlow\.js.

✅ Example:

```python
import tensorflow as tf
tf.saved_model.save(model, "exported_model")
```

---

### 6. **HDF5 (`.h5`) \[Keras legacy format]**

* **What it is:**
  HDF5-based storage for Keras models (before TensorFlow 2.x standardized SavedModel).
* **Pros:**

  * Human-readable metadata.
  * Easy to share.
* **Cons:**

  * Mostly legacy — SavedModel preferred now.
* **Use Cases:**

  * Saving Keras models in older projects.

---

# 📊 Quick Comparison

| Format        | Ecosystem       | Portability | Security Risk | Deployment Fit             | Best For                  |
| ------------- | --------------- | ----------- | ------------- | -------------------------- | ------------------------- |
| Pickle/Joblib | Python only     | ❌ Low       | ⚠️ High       | Dev/prototyping            | Quick saves in Python     |
| ONNX          | Multi-framework | ✅ High      | ✅ Safer       | ✅ Production (multi-stack) | Cross-platform deployment |
| TorchScript   | PyTorch         | Medium      | ✅ Safer       | ✅ PyTorch Serving, Mobile  | PyTorch production        |
| TF SavedModel | TensorFlow      | Medium-High | ✅ Safer       | ✅ TF Serving, TFLite, JS   | TensorFlow production     |
| HDF5 (.h5)    | Keras legacy    | Medium      | ⚠️ Moderate   | Legacy only                | Older Keras               |

---

👉 So, in MLflow Registry, you’ll often see:

* **Pickle/Joblib** for scikit-learn.
* **ONNX** for portable models.
* **TorchScript** for PyTorch.
* **SavedModel** for TensorFlow.

---

Would you like me to also explain **which of these formats are most vulnerable** (e.g., pickle risks) and how to **harden them inside MLflow pipelines**?
