# Data
## Mean
## Median
## Mode
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
### Gaussian / Normal 
### 



# Linear Regression
- Supervised Learning

## Training
- Prepare the dataset
  - Features: inputs (X)
  - Labels: target values (y)

- Initialize model parameters
  - Weights / coefficients
  - Bias (intercept)

- Prediction–Loss–Update Cycle
  - Forward pass: use linear equation (y = wX + b) to generate predictions
  - Compute loss using loss function (e.g., MSE, with optional regularization)
  - Backward pass (backpropagation):
    - Compute gradients using partial derivatives of loss w.r.t. weights and bias
  - Apply solver (e.g., Gradient Descent) to update parameters
  - Repeat for multiple iterations (epochs)

- Monitor training
  - Track loss value
  - Check convergence

- Stop training
  - When loss stabilizes or reaches acceptable minimum


  
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

 
 


