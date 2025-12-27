
# Classification

Classification predicts a category or label (e.g., spam/not spam, cat/dog) from a set of options, while Regression predicts a continuous numerical value (e.g., price, temperature, age) within a range, both learning patterns from labeled data to make future predictions. The key difference is the output type: discrete for classification, continuous for regression. 


| Situation                        | Best Classifier(s)                                   |
| -------------------------------- | ---------------------------------------------------- |
| **Text / NLP**                   | Naïve Bayes, Logistic Regression, Transformers       |
| **Images**                       | CNNs, SVM with kernels                               |
| **Tabular Data**                 | Random Forest, Gradient Boosting (XGBoost, LightGBM) |
| **Time Series / Speech**         | RNNs, LSTMs, HMMs                                    |
| **Small dataset, fast training** | Logistic Regression, Naïve Bayes                     |
| **Complex, high accuracy**       | Gradient Boosting, Deep Learning                     |

###  Classification Paradigms 

| **Aspect**                                       | **Binary Classification**                                                | **Multi-Class Classification**                                             | **Multi-Label Classification**                                                                      | **Hierarchical Multi-Label Classification (HMC)**                                                               |
| ------------------------------------------------ | ------------------------------------------------------------------------ | -------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| **Definition**                                   | Predict **one of two** possible classes.                                 | Predict **one class** out of **three or more** mutually exclusive classes. | Predict **one or more classes simultaneously** (non-mutually exclusive).                            | Predict **one or more labels**, organized in a **class hierarchy/tree structure** (parent-child relationships). |
| **Output type**                                  | Single binary output: {0,1} or {Yes,No}.                                 | One categorical output (e.g., {A,B,C,...}).                                | A **binary vector** (e.g., [1,0,1,0]) — each element represents presence/absence of label.          | A **structured label vector/tree** — includes dependencies (e.g., selecting “Dog” implies “Animal”).            |
| **Model output layer**                           | 1 neuron with **sigmoid** activation.                                    | N neurons with **softmax** activation (probabilities sum to 1).            | N neurons with **sigmoid** activation (independent probabilities per label).                        | N neurons with **sigmoid** + constraint mechanism (enforces hierarchical consistency).                          |
| **Loss function**                                | Binary Cross-Entropy (Log Loss).                                         | Categorical Cross-Entropy.                                                 | Binary Cross-Entropy (per label).                                                                   | Hierarchical loss (e.g., weighted BCE or hierarchical cross-entropy that penalizes parent-child violations).    |
| **Evaluation metrics**                           | Accuracy, Precision, Recall, F1-score, ROC-AUC.                          | Accuracy, Macro/Micro-F1, Confusion Matrix.                                | Hamming Loss, Precision@k, Recall@k, F1-micro/macro, Subset Accuracy.                               | Hierarchical Precision/Recall/F1, Tree-induced loss, Path-based accuracy.                                       |
| **Independence assumption**                      | Two classes are **mutually exclusive**.                                  | Classes are **mutually exclusive** (only one correct).                     | Labels are **independent** — multiple may be correct.                                               | Labels are **dependent** — follow a **taxonomy or DAG** (Directed Acyclic Graph).                               |
| **Example**                                      | Spam vs. Not Spam email.                                                 | Predicting type of animal: {cat, dog, horse}.                              | Predicting movie genres: {action, comedy, drama}.                                                   | Predicting news topic hierarchy: {Politics → Election → US Presidential}.                                       |
| **Problem formulation**                          | ( y \in {0,1} )                                                          | ( y \in {1,2,...,K} )                                                      | ( y \in {0,1}^K )                                                                                   | ( y \in {0,1}^K ) with hierarchical constraints                                                                 |
| **Training data requirement**                    | Balanced or reweighted binary samples.                                   | Large, diverse labeled data covering all classes.                          | Each instance annotated with multiple labels.                                                       | Each instance labeled with multiple levels of hierarchy.                                                        |
| **Model examples**                               | Logistic Regression, SVM (binary), Decision Tree, Neural Net (1 output). | Softmax Regression, CNN classifier, Random Forest, BERT fine-tuning.       | Sigmoid-based neural network, Problem Transformation methods (Binary Relevance, Classifier Chains). | Graph Neural Networks, Hierarchical Attention Networks, Ontology-based classifiers.                             |
| **Decision boundary**                            | Linear or nonlinear separating 2 classes.                                | Multiple decision regions partitioning feature space.                      | Multiple overlapping decision boundaries — one per label.                                           | Hierarchy-aware decision boundaries (must satisfy parent-child rules).                                          |
| **Complexity**                                   | O(1) classifier.                                                         | O(K) classes → single classifier handling all.                             | O(K) binary sub-problems or one large sigmoid network.                                              | O(K) but hierarchical dependency increases computational complexity.                                            |
| **Interpretability**                             | High (easy to visualize).                                                | Moderate.                                                                  | Moderate–low (multi-dimensional outputs).                                                           | Low–complex, but interpretable via tree structure.                                                              |
| **Typical libraries / frameworks**               | scikit-learn (`LogisticRegression`)                                      | scikit-learn, TensorFlow/Keras softmax models                              | scikit-learn’s `MultiOutputClassifier`, `BinaryRelevance` from `scikit-multilearn`                  | `sklearn-hierarchical`, `keras-hmc`, ontology-based ML pipelines                                                |
| **Use cases**                                    | Credit approval, disease diagnosis (positive/negative).                  | Image classification, language identification.                             | Tag recommendation, music genre detection, toxic comment classification.                            | Document categorization with taxonomy (e.g., Wikipedia categories, product catalogs).                           |
| **Shortcomings**                                 | Limited to 2 outcomes.                                                   | Cannot assign multiple valid labels.                                       | Ignores label correlations unless modeled.                                                          | Complex to train and evaluate; label imbalance across hierarchy.                                                |
| **Improvement over predecessor**                 | —                                                                        | Extends binary to >2 classes.                                              | Removes exclusivity assumption.                                                                     | Adds semantic structure and dependency awareness.                                                               |
| **Example output (for document classification)** | “Spam”                                                                   | “Sports”                                                                   | “Sports”, “Health”                                                                                  | “News → Sports → Cricket”                                                                                       |
| **Common model architecture**                    | Single sigmoid neuron                                                    | Softmax dense layer                                                        | Multi-sigmoid output layer                                                                          | Multi-sigmoid + constraint propagation network                                                                  |
| **Evaluation difficulty**                        | Simple                                                                   | Moderate                                                                   | Hard (multi-dimensional)                                                                            | Very hard (hierarchical structure, dependency penalties)                                                        |





```
Binary Classification
     ↓
Multi-Class Classification
     ↓
Multi-Label Classification
     ↓
Hierarchical Multi-Label Classification
```



Each step generalizes the previous one:

* Binary → only 2 labels
* Multi-class → more than 2, but mutually exclusive
* Multi-label → multiple independent labels
* Hierarchical → multiple dependent labels (tree/DAG relationships)



| Scenario                          | Classification Type      | Example                       |
| --------------------------------- | ------------------------ | ----------------------------- |
| Email: Spam or Not                | Binary                   | Spam detection                |
| Image: Cat / Dog / Bird           | Multi-Class              | Vision classifier             |
| Song: Rock + Jazz + Blues         | Multi-Label              | Music genre tagging           |
| News: World → Politics → Election | Hierarchical Multi-Label | Taxonomy-based classification |
 
 In **Machine Learning** there are many **classifiers** (algorithms that assign input data to a class/label).






A **classifier** is a machine learning model (or algorithm) that **assigns input data to a category (class)**.

* If the output is **discrete (categorical)** → It’s a **classification problem**.
* If the output is **continuous (numeric)** → That’s **regression**, not classification.

 Example:

* Input: an email
* Classifier Output: **Spam** or **Not Spam**

So the classifier is the model that **learns from training data** and then **predicts the class of new data**.



###  Types of Classifiers

1. **Binary Classifier**

   * Only two classes.
   * Example: Fraud detection (**fraud / not fraud**).

2. **Multiclass Classifier**

   * More than two classes.
   * Example: Image classification (**cat, dog, bird, car**).

3. **Multilabel Classifier**

   * Each input can belong to **multiple classes simultaneously**.
   * Example: Movie classification (**Action + Comedy + Romance**).

 

### Classifier Algorithms

| Algorithm                           | Description                            | Example Use Case           |
| ----------------------------------- | -------------------------------------- | -------------------------- |
| **Logistic Regression**             | Linear model for binary classification | Spam detection             |
| **Decision Tree**                   | Tree-based rules for classification    | Loan approval              |
| **Random Forest**                   | Ensemble of trees                      | Medical diagnosis          |
| **Naive Bayes**                     | Probabilistic classifier               | Text classification        |
| **SVM (Support Vector Machine)**    | Finds best boundary between classes    | Face recognition           |
| **k-NN (k Nearest Neighbors)**      | Uses neighbors to classify             | Recommender systems        |
| **Neural Networks / Deep Learning** | Multi-layer perceptrons, CNNs, RNNs    | Image & speech recognition |

 

###  Example of a Classifier

Suppose we build a **classifier for exam results** based on:

* Hours studied
* Sleep quality

Training data:

| Hours Studied | Sleep Quality | Result |
| ------------- | ------------- | ------ |
| 5             | 0.9           | Pass   |
| 1             | 0.5           | Fail   |
| 4             | 0.7           | Pass   |
| 2             | 0.3           | Fail   |

 The classifier learns a **decision boundary**.

* New input: **\[3 hours, 0.8 sleep]** → Classifier predicts **Pass**.

 

So, in short:
A **classifier = ML model that assigns categories to inputs**.

  

###  1. **Probabilistic Classifiers**

These are based on probability theory.

| Classifier                  | Idea                                                  | Example Use Cases                       |
| --------------------------- | ----------------------------------------------------- | --------------------------------------- |
| **Naïve Bayes**             | Assumes features are independent, uses Bayes’ theorem | Spam filtering, sentiment analysis      |
| **Gaussian Naïve Bayes**    | Assumes features follow Gaussian distribution         | Medical diagnosis                       |
| **Multinomial Naïve Bayes** | Works well for text & word counts                     | Document classification                 |
| **Bernoulli Naïve Bayes**   | For binary features (yes/no, 0/1)                     | Text classification (presence of words) |
| **Bayesian Networks**       | Graph-based conditional probability                   | Fraud detection, gene networks          |



###  2. **Linear Classifiers**

These separate classes using linear decision boundaries.

| Classifier                             | Idea                                       | Example Use Cases            |
| -------------------------------------- | ------------------------------------------ | ---------------------------- |
| **Logistic Regression**                | Uses sigmoid to output class probabilities | Credit risk prediction       |
| **Linear Discriminant Analysis (LDA)** | Maximizes class separability               | Face recognition, biometrics |
| **Perceptron**                         | Oldest neural network, linear separator    | Pattern recognition          |



###  3. **Tree-Based Classifiers**

These split data based on feature conditions.

| Classifier                                               | Idea                                 | Example Use Cases                           |
| -------------------------------------------------------- | ------------------------------------ | ------------------------------------------- |
| **Decision Tree**                                        | Splits data into rules (if-else)     | Customer churn prediction                   |
| **Random Forest**                                        | Ensemble of many decision trees      | Fraud detection, healthcare                 |
| **Gradient Boosted Trees (XGBoost, LightGBM, CatBoost)** | Sequential trees that correct errors | Kaggle competitions, recommendation systems |



###  4. **Instance-Based Classifiers**

These compare test samples directly to training samples.

| Classifier                               | Idea                                             | Example Use Cases       |
| ---------------------------------------- | ------------------------------------------------ | ----------------------- |
| **K-Nearest Neighbors (KNN)**            | Classifies by majority vote of nearest neighbors | Handwriting recognition |
| **Kernel Density Estimation Classifier** | Uses probability density                         | Anomaly detection       |



###  5. **Support Vector Machines (SVM)**

| Variant                         | Idea                            | Example Use Cases   |
| ------------------------------- | ------------------------------- | ------------------- |
| **Linear SVM**                  | Finds maximum-margin hyperplane | Text classification |
| **Non-linear SVM (Kernel SVM)** | Uses kernels (polynomial, RBF)  | Image recognition   |



###  6. **Neural Network Classifiers**

| Type                                              | Idea                               | Example Use Cases           |
| ------------------------------------------------- | ---------------------------------- | --------------------------- |
| **Multilayer Perceptrons (MLP)**                  | Feedforward deep networks          | Tabular data classification |
| **Convolutional Neural Networks (CNNs)**          | Capture spatial patterns           | Image classification        |
| **Recurrent Neural Networks (RNNs, LSTMs, GRUs)** | Capture sequence/time dependencies | Speech, NLP                 |
| **Transformers (BERT, GPT, etc.)**                | Self-attention based               | NLP, vision-language tasks  |



###  7. **Ensemble Methods**

These combine multiple classifiers to improve performance.

| Classifier                                 | Idea                                      | Example Use Cases    |
| ------------------------------------------ | ----------------------------------------- | -------------------- |
| **Bagging** (Bootstrap Aggregating)        | Averages predictions of multiple models   | Reduces variance     |
| **Boosting** (AdaBoost, Gradient Boosting) | Sequentially improves weak learners       | Fraud detection      |
| **Stacking**                               | Combines outputs of different classifiers | Complex competitions |



###  8. **Other Specialized Classifiers**

| Classifier                          | Idea                                          | Example Use Cases        |
| ----------------------------------- | --------------------------------------------- | ------------------------ |
| **Hidden Markov Models (HMMs)**     | Probabilistic sequence classifier             | Speech recognition       |
| **k-Means + Label Assignment**      | Clustering turned into classification         | Market segmentation      |
| **Fuzzy Classifiers**               | Assigns probabilities instead of hard classes | Medical diagnostics      |
| **Probabilistic Neural Nets (PNN)** | Distance-based neural classifiers             | Biometric authentication |






* Naïve Bayes is one of the simplest classifiers (probabilistic).
* But there are **linear, tree-based, distance-based, SVM, neural networks, and ensemble classifiers**.
* Choice depends on data type (text, image, sequence, tabular) and trade-off (accuracy vs interpretability vs speed).

