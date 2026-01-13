Below is a **compact AI-engineer–oriented list of common loss functions**, grouped by use case, with **one-line explanations**.

---

## **Regression Losses**

* **Mean Squared Error (MSE)** – Penalizes squared differences; strongly punishes large errors.
* **Mean Absolute Error (MAE)** – Penalizes absolute differences; robust to outliers.
* **Root Mean Squared Error (RMSE)** – Square root of MSE; error in original units.
* **Huber Loss** – Combines MSE (small errors) and MAE (large errors).
* **Log-Cosh Loss** – Smooth version of MAE; less sensitive to outliers.
* **Quantile Loss** – Used for predicting conditional quantiles instead of mean.

---

## **Classification Losses**

* **Binary Cross-Entropy (Log Loss)** – Measures error for binary classification using probabilities.
* **Categorical Cross-Entropy** – Used for multi-class classification with one-hot labels.
* **Sparse Categorical Cross-Entropy** – Same as categorical CE but with integer labels.
* **Hinge Loss** – Used in SVMs; penalizes predictions inside the margin.
* **Squared Hinge Loss** – Smooth variant of hinge loss.
* **Focal Loss** – Down-weights easy examples to focus on hard ones (class imbalance).

---

## **Probabilistic / Information-Theoretic Losses**

* **Negative Log Likelihood (NLL)** – Penalizes low likelihood of the true class.
* **KL Divergence** – Measures how one probability distribution diverges from another.
* **Jensen–Shannon Divergence** – Symmetric and bounded version of KL divergence.

---

## **Ranking & Recommendation Losses**

* **Contrastive Loss** – Pulls similar samples together and pushes dissimilar ones apart.
* **Triplet Loss** – Enforces distance between anchor–positive and anchor–negative samples.
* **Cosine Embedding Loss** – Optimizes similarity based on cosine distance.
* **Pairwise Ranking Loss** – Optimizes relative ordering between pairs.

---

## **Computer Vision Losses**

* **IoU Loss** – Penalizes poor overlap between predicted and true bounding boxes.
* **GIoU / DIoU / CIoU Loss** – Improved IoU variants considering distance and shape.
* **Dice Loss** – Measures overlap for segmentation tasks.
* **Tversky Loss** – Generalized Dice loss for imbalanced segmentation.

---

## **Sequence & NLP Losses**

* **Sequence Cross-Entropy Loss** – Cross-entropy summed over all time steps.
* **CTC Loss** – Aligns input and output sequences without explicit alignment.
* **Language Modeling Loss** – Cross-entropy over next-token prediction.

---

## **Generative Model Losses**

* **GAN Loss** – Adversarial loss between generator and discriminator.
* **Wasserstein Loss** – Stable GAN loss using Earth Mover’s distance.
* **ELBO Loss** – VAE loss combining reconstruction error and KL divergence.

---

## **One-line interview summary**

> **Loss functions quantify prediction error, guide optimization, and are chosen based on task type—regression, classification, ranking, vision, sequence, or generative modeling.**

If you want, I can also give:

* **“Which loss to use when” table**, or
* **Top 10 most commonly used losses for interviews**
