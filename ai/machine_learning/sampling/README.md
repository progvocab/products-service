Sampling in AI is the process of **selecting outputs from a probability distribution** instead of always choosing the most likely one.
It introduces **controlled randomness**, making model outputs more diverse and natural.
Common sampling methods include **greedy, temperature, top-k, and top-p sampling**.

Below are the **common sampling methods used in AI (especially in generative models)** with a one-line explanation for each:

* **Greedy Sampling** – Always picks the token with the highest probability.
* **Random Sampling** – Samples directly from the full probability distribution.
* **Temperature Sampling** – Controls randomness by scaling probabilities.
* **Top-K Sampling** – Samples only from the K most probable tokens.
* **Top-P (Nucleus) Sampling** – Samples from the smallest set of tokens whose cumulative probability ≥ P.
* **Beam Search** – Explores multiple likely sequences and selects the best overall.
* **Typical Sampling** – Prefers tokens with probability close to the expected distribution entropy.
* **Contrastive Search** – Balances relevance and diversity by penalizing repetition.

These methods trade off **accuracy, diversity, and determinism** depending on the use case.



What is sampling and why it is used

Sampling = selecting a subset of data from a larger dataset

It is used to:

1. Reduce computation cost (training faster, less memory)


2. Handle class imbalance


3. Reduce variance (used in ensembles like Random Forest, XGBoost)


4. Enable faster experimentation


5. Create train/validation splits




---

Common types of sampling (with intuition)

1️⃣ Random sampling

Pick rows randomly.

Use when: dataset is very large but representative.


---

2️⃣ Stratified sampling

Preserve class proportions.

Use when: classification with imbalanced classes.


---

3️⃣ Sampling with replacement (Bootstrap)

Same row can appear multiple times.

Use when: bagging, Random Forest.


---

4️⃣ Row subsampling (used in XGBoost)

Each tree sees only a fraction of rows.

Use when: reduce overfitting (variance).


---

5️⃣ Column sampling

Each model uses only a subset of features.

Use when: decorrelate trees.


---

Code snippets (Pandas DataFrame)

Assume:

import pandas as pd

1️⃣ Simple random sampling

sample_df = df.sample(frac=0.2, random_state=42)  # 20% data


---

2️⃣ Random sampling with fixed size

sample_df = df.sample(n=1000, random_state=42)


---

3️⃣ Stratified sampling (classification)

from sklearn.model_selection import train_test_split

X = df.drop("target", axis=1)
y = df["target"]

X_sample, _, y_sample, _ = train_test_split(
    X, y,
    train_size=0.2,
    stratify=y,
    random_state=42
)


---

4️⃣ Sampling with replacement (Bootstrap)

bootstrap_df = df.sample(frac=1.0, replace=True, random_state=42)


---

5️⃣ Train / Validation split

train_df = df.sample(frac=0.8, random_state=42)
val_df = df.drop(train_df.index)


---

Sampling in ML models (important)

You often don’t sample manually; models do it internally.

Example: XGBoost row sampling

XGBClassifier(
    subsample=0.8,
    colsample_bytree=0.8
)

subsample → row sampling

colsample_bytree → feature sampling



---

When NOT to sample

❌ Very small datasets
❌ When rare but important cases may be lost
❌ Time-series data (random sampling breaks order)


---

One-line summary

✅ Sampling reduces cost, controls overfitting, and enables scalable ML; in pandas it’s done using df.sample() or stratified splits.

If you want, I can next:

Show sampling for imbalanced data (SMOTE vs undersampling)

Explain why sampling reduces variance mathematically

Compare sampling vs regularization

Show sampling in Spark / Glue
