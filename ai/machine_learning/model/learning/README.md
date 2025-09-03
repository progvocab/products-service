

---

# 🔹 1. **Supervised Learning**

* **Idea**: The model learns from **labeled data** (input → correct output given).
* **Goal**: Minimize the error between predictions and actual labels.
* **How it learns**:

  * You feed the model input $x$ and correct output $y$.
  * The model makes a prediction $\hat{y}$.
  * Compute **loss** (e.g., mean squared error, cross-entropy).
  * Use **gradient descent** to update parameters.

✅ **Examples**:

* Predicting house prices (input: features, output: price).
* Image classification (input: picture, output: cat/dog).

---

# 🔹 2. **Unsupervised Learning**

* **Idea**: The model learns patterns **without labels**.
* **Goal**: Discover hidden structure in data.
* **How it learns**:

  * Input $x$ only (no $y$).
  * The model groups or compresses the data.

✅ **Examples**:

* Clustering customers (K-means, DBSCAN).
* Dimensionality reduction (PCA, autoencoders).
* Market basket analysis (finding frequent item sets).

---

# 🔹 3. **Reinforcement Learning (RL)**

* **Idea**: An **agent** learns by interacting with an **environment**.
* **Goal**: Maximize **cumulative reward** over time.
* **How it learns**:

  * At each step, agent observes **state $s$**.
  * Chooses an **action $a$**.
  * Gets a **reward $r$** and moves to new state.
  * Updates its **policy** to improve long-term rewards.

✅ **Examples**:

* AlphaGo (learning to play Go).
* Self-driving cars.
* Robotics.

---

# 🔹 4. **Semi-Supervised Learning**

* **Idea**: Combination of labeled and unlabeled data.
* **Goal**: Improve performance when labeled data is scarce.
* **How it learns**:

  * Small portion of labeled data guides the model.
  * Large unlabeled data helps the model generalize.

✅ **Examples**:

* Medical imaging (few labeled scans, many unlabeled scans).

---

# 🔹 5. **Self-Supervised Learning**

* **Idea**: Labels are generated automatically from the data itself.
* **Goal**: Learn representations without manual labeling.
* **How it learns**:

  * Predict a part of the input from other parts.
  * Example: Masked Language Modeling in Transformers (predict missing word).

✅ **Examples**:

* GPT, BERT (predict next/missing word).
* Vision models (predict missing patch in an image).

---

# 🔹 6. **Online Learning**

* **Idea**: Model learns **incrementally** as data arrives.
* **Goal**: Adapt to streaming or continuously changing data.

✅ **Examples**:

* Stock price prediction.
* Recommendation systems adapting to new users.

---

# 🔹 Comparison Table

| Type                | Data Needed     | Labels?               | Goal                       | Examples                   |
| ------------------- | --------------- | --------------------- | -------------------------- | -------------------------- |
| **Supervised**      | Fixed dataset   | ✅ Yes                 | Predict known outputs      | Classification, regression |
| **Unsupervised**    | Fixed dataset   | ❌ No                  | Find structure             | Clustering, PCA            |
| **Reinforcement**   | Sequential env  | ❌ No (reward instead) | Maximize reward            | Games, robotics            |
| **Semi-Supervised** | Both            | Partially             | Improve with little labels | Medical, speech            |
| **Self-Supervised** | Large unlabeled | ❌ No (auto-label)     | Pretrain representations   | GPT, BERT                  |
| **Online**          | Streaming data  | Either                | Learn incrementally        | Real-time systems          |

---

⚡ Would you like me to also **draw a visual flow diagram** showing how data flows in each learning type (e.g., supervised vs RL vs unsupervised)?
