# Data
## Mean
## Median
## Mode
- Bimodal
- Multimodal 
## Range
## Deviation 
- Simple
- Absolute 
- Variance 
- Standard Deviation 
- Median Absolute Deviation 
- Gaussian Distribution 
## Z score

## Skewness
- positive 
- negative 
- symmetrical 

## Random variable 
- Discrete 
- Continuous 
## Probability 
- Event 
- Outcome 
- Conditional Probability 

# Probability Distribution 
## Discrete 
- Probability Mass Function 
### Bernoulli
- 1 trial
- Output is 0 or 1
- Probability of Success 
- Probability of Failure 


### Binomial 
- n trial
- Output is 0 to n
- sum of n independent Bernoulli variables 
- count success in fixed n trails

### Poisson 
- n trails , very large n
- very small Probability of Success 
- count rare events in fixed time space interval 
- rare event approximation of Binomial

## Continuous 
- Probability Density Function 
### Gaussian / Normal 
### 

# Supervised Learning 
- Stochastic 
  - Random sampling 
  - different results for same input 
- Probabilistic 
  - Result is a probability 
  - Cat : 85 % , Dog : 15%
- Deterministic 
  - Result is a single class 
  - Same input produces same Result on each execution 


# Naive Bayes 
- assumes features are independent , hence naive
- calculates probability of each class 
- predicts one with highest probability 


## Bayes Theorem 

- class Conditional independence 
- class prior independence

# Linear Regression
- Supervised Learning
- assumes Linear Relationship 

## Training
- Prepare the dataset
  - Features: inputs (X)
  - Labels: target values (y)

- Initialize model parameters
  - Weights / coefficients
  - Bias (intercept)


- Forward Computation use linear equation (y = wX + b) to generate predictions
  - Compute loss using loss function
    - MSE, with optional regularization
  
- Solve Optimization Problem 
  - apply objective function to define goal
    - Ordinary Least Square to minimize MSE
- Apply solver to update parameters
    - Compute parameters using matrix decomposition 
      - SVD Singular Value Decomposition 
        - divides features metrix into 3 metrics for 
        - numerical stability
        - psuedo inverse computation 
        - handles multi collinearity
        - Dimensionality Reduction 
        - Geometric Insight 
          - Direction ( First and third matrix)
          - Importance ( second )
          - Rotation Scaling 
        - Eigen vector is a non zero matrix when multiplied by a Square Matrix,  it does not change direction
        - Eigen values are factors by which Eigen vectors are stretch during transformations 
        - Compute weights using Ordinary Least Square 
      - QR
- Parameters obtained in single shot

- Monitor training
  - Track loss value
  - Check convergence

- Stop training
  - When loss stabilizes or reaches acceptable minimum

## Bayesian Linear Regression 
- Regression coefficients as Probability Distribution instead of fixed point estimates.
- Gaussian Distribution for components 
  - Prior 
  - Likelihood
  - Posterior 

## Ridge Regression 
- Apply penalty to coefficients 
- Solvers
- lbfgs 
  - Quasi-Newton solver 
  - uses Gradient information and approximate the Hessian to minimize loss
  
- cholesky
  - Ordinary Least Square using Cholesky factorization 

- sparse_cg 
  - iterative Conjugate Gradient Solvers without matrix inverse 

## Stochastic Gradient Descent 

- Backward pass (backpropagation):
    - Compute gradients using partial derivatives of loss w.r.t. weights and bias

# Support Vector Machines 

- Kernel Trick
  - enables handling non linearly separable data without explicitly computing high dimension feature mapping 
  - simplifies computation by replacing high dimension dot product with Kernel function 
  - instead of transforming and then computing similarity,  Kernel functions directly calculate results.


### SVM Training Process
- Problem Setup
  - Classification or Regression
  - Binary or Multi-class
  - Choose kernel type

- Data Preparation
  - Feature scaling 
    - Standardization
    - Normalization
  - Handle missing values
  - Train–test split

- Choose Kernel
  - Linear
  - Polynomial
  - RBF
  - Sigmoid
  - Precomputed

- Optimization Objective
  - Maximize margin
  - Minimize hinge loss
  - Soft margin formulation
    - Slack variables (ξ)
  - Regularization parameter (C)

- Hyperparameters
  - C (regularization strength)
  - gamma (RBF/poly/sigmoid)
  - degree (polynomial)
  - coef0 (poly/sigmoid)

- Optimization Solver
  - Quadratic Programming (QP)
  - SMO (Sequential Minimal Optimization)
  - Convergence criteria
- Support Vectors
  - Points on margin
  - Define decision boundary
  - Sparse solution
- Model Evaluation
  - Accuracy / Precision / Recall
  - Cross-validation
    - K fold
    - Stratified K fold
    - Leave one out
    - Shuffle 
  - Hyperparameter tuning 
    - GridSearch 
    - RandomSearch
- Final Model
    - Decision function
    - Predict new samples
    - Deployment


### Kernels Functions 

Transform lower dimension non linearly separable data into 
higher dimension spaces where Linear separator ( hyperplane ) can be found.
-  Linear Kernel
  - Formula: K(x, y) = x · y
  - No feature transformation
- Polynomial Kernel
  - Formula: K(x, y) = (γ x·y + r)^d
  - Parameters:
    - degree (d)
    - gamma (γ)
    - coef0 (r)
  - Captures feature interactions
  - Use case:
    - Non-linear but structured data
- RBF (Radial Basis Function) Kernel
  - Formula: K(x, y) = exp(-γ ||x - y||²)
  - Most commonly used
  - Parameter:
    - gamma (γ)
  - Maps to infinite-dimensional space
  - Use case:
    - General-purpose non-linear classification
- Sigmoid Kernel
  - Formula: K(x, y) = tanh(γ x·y + r)
  - Related to neural networks
  - Parameters:
    - gamma
    - coef0
  - Less commonly used
- Precomputed Kernel
  - User provides custom kernel matrix
  - Enables:
    - Graph kernels
    - String kernels
    - Domain-specific similarity
  - Advanced use cases



## Decision Tree 
- Pruning 
  - Pre pruning 
    - stop the tree from growing during training 
    - constraints 
      - max depth
      - min samples
    - prevents overfitting 
    - prone to underfitting 
  - Post pruning 
    - grow full tree later cut unnecessary branches 
    - Techniques 
      - Cost complexity pruning 
      - Reduced error pruning 

# Apriori 
- Association Rule Mining 
- identity frequent patterns,  connection,  dependency 
- Antecedent 
  - first item or itemset
- Consequent
  - second item
## Steps
- frequent itemset generation 
- expand and prune itemset
- Repeat steps 1 and 2
## Itemset 
- different group of items
- Measure using
  - Support 
  - Confidence 
  - Lift

## Loss functions 
 

### Mean Squared Error (MSE)
- Calculate error for each data point (actual − predicted).
- Square each error to penalize larger errors more.
- Compute the average of squared errors across all data points.
- (Optional) Add regularization term to the loss (Ridge/Lasso).

### Root Mean Squared Error (RMSE)
### Mean Absolute Error (MAE)
### Huber Loss
### Log-Cosh Loss
### Quantile Loss
### Mean Squared Logarithmic Error (MSLE)
### Poisson Loss
### Tweedie Loss
### Mean Absolute Percentage Error (MAPE)
### Symmetric Mean Absolute Percentage Error (SMAPE)


## Solver
### Normal Equation (closed-form)
### Batch Gradient Descent
### Stochastic Gradient Descent (SGD)
### Mini-Batch Gradient Descent
### Coordinate Descent
### Conjugate Gradient
### L-BFGS
### Newton’s Method
### Gauss-Newton
### Levenberg–Marquard
 
## Regularization

### Lasso Regression

**L1 regularization** adds the absolute value of coefficients to the loss function.

> Can shrink some coefficients to **exactly zero**, which performs **feature selection**.
*  lambda  is the **regularization strength**
  *  controls how much penalty is applied.
  * Larger  lambda  → more coefficients shrink to zero.

 

 

### Ridge Regression

**L2 regularization** adds the squared value of coefficients to the loss function.

> Shrinks coefficients toward zero but **never exactly zero**. Helps reduce **overfitting** and **multicollinearity**.
* Larger  lambda   → stronger shrinkage.



```shell

Ensemble Learning
│
├── 1️⃣ Bagging (Bootstrap Aggregating)
│   ├── Train models independently
│   ├── Random sampling with replacement
│   ├── Reduces variance
│   ├── Parallel training
│   └── Example:
│        └── Random Forest
│
├── 2️⃣ Boosting
│   ├── Train models sequentially
│   ├── Each model corrects previous errors
│   ├── Reduces bias (and variance)
│   ├── Weighted data updates
│   └── Examples:
│        ├── AdaBoost
│        ├── Gradient Boosting
│        ├── XGBoost
│        ├── LightGBM
│        └── CatBoost
│
├── 3️⃣ Stacking (Stacked Generalization)
│   ├── Train multiple base models
│   ├── Predictions used as features
│   ├── Meta-model learns final output
│   ├── Combines heterogeneous models
│   └── Example:
│        └── Logistic Regression as meta-learner
│
└── 4️⃣ Voting
    ├── Train multiple models
    ├── Combine predictions directly
    ├── Simple aggregation
    ├── Types:
    │     ├── Hard Voting (majority class)
    │     └── Soft Voting (average probabilities)
    └── Works well with diverse models

```


# Uncertainty Quantization 
- Model level Uncertainty 
- Data level Uncertainty 

## Posterior Variance 
- Change in model parameters After seeing the data
- Model disagreement 
### Ensemble Methods 

### Bayesian Linear Regression 

### Monte Carlo Sampling 

### Monte Carlo Dropout 



## Predictive Variance 
- Change in model prediction 


# Semi Supervised Learning 
- Labelled and Unlabelled Data 
## Assumptions 
### Cluster
Unlabelled Data point is assigned the same class as it's nearest Labelled Data point.
### Smoothness
Unlabelled Data point is assigned the same label as it's nearest Labelled Data point.
### Low Density 
Decision boundary should not pass through high Density region 
### Manifold
High dimension input space comprises multiple  low dimension Manifold 


### Transductive Learning 
- Process of drawing conclusion about the new data from previous without creation of a model

### Inductive Learning 
- tries to create a general rule for a specific data

### Label Propagation 
- graph based algorithm tries to compute labels for Unlabelled Data based on their proximity to Labelled Data 
- used Euclidean distance 

