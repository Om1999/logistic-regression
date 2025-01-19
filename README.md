# Logistic Regression Implementation with Assumption Testing

## Overview
This repository contains a practical implementation of **Logistic Regression** from scratch, using Python. The goal of this project is to demonstrate the application of logistic regression and to test the assumptions behind it, such as **linearity**, **multicollinearity**, and **outlier influence**. The code also includes functions for **cost function calculation**, **gradient descent**, and **decision boundary plotting**.

## Prerequisites
To run this code, you will need to have the following libraries installed:

- **numpy**: For numerical operations.
- **matplotlib**: For plotting visualizations.
- **seaborn**: For enhanced plotting styles.
- **sklearn**: For generating datasets and using logistic regression tools.
- **statsmodels**: For assumption testing and statistical modeling.

You can install these libraries using `pip`:
```
## Project Structure
```
logistic_regression_project/
│
├── logistic_regression.ipynb      # Main code for logistic regression and assumption testing
├── README.md                      # Project documentation
```

## Functions

### 1. `generate_data()`
Generates a synthetic dataset for classification using `make_classification` from sklearn.

**Usage**:
```python
X, y = generate_data()
```

**Returns**:
- `X`: Feature matrix (5000 samples, 2 features).
- `y`: Class labels (binary: 0 or 1).

---

### 2. `add_intercept(X)`
Adds a column of ones to the feature matrix `X` to account for the bias (intercept) in logistic regression.

**Usage**:
```python
XX = add_intercept(X)
```

---

### 3. `sigmoid(z)`
Calculates the sigmoid function of the input `z`, returning predicted probabilities.

**Usage**:
```python
h = sigmoid(z)
```

---

### 4. `calc_h(X, theta)`
Calculates the hypothesis (predicted probabilities) using the logistic regression model.

**Usage**:
```python
h = calc_h(X, theta)
```

---

### 5. `gradient_descent(X, y, theta, alpha, num_iter)`
Performs gradient descent to minimize the cost function and find the optimal parameters for logistic regression.

**Usage**:
```python
cost_list, optimal_parameters = gradient_descent(X, y, theta, alpha, num_iter)
```

**Parameters**:
- `alpha`: Learning rate.
- `num_iter`: Number of iterations.

---

### 6. `logistic_regression(X, y, alpha=0.01, num_iter=100000)`
Runs logistic regression on the dataset, using gradient descent to find the optimal parameters and returns the cost history and optimal parameters.

**Usage**:
```python
cost_list, optimal_parameters = logistic_regression(X, y)
```

---

### 7. `plot_decision_boundary(X, y, theta)`
Plots the decision boundary of the logistic regression model along with the data points.

**Usage**:
```python
plot_decision_boundary(X, y, optimal_parameters)
```

---

### 8. `assumption_testing(X, y, theta)`
Performs assumption testing for logistic regression, including tests for:
- Linearity of independent variables and log odds.
- Influential outliers using Cook's Distance.
- Multicollinearity using Variance Inflation Factor (VIF).
- Sample size sufficient for logistic regression.

**Usage**:
```python
assumption_testing(X, y, optimal_parameters)
```

## Usage

### Step 1: Generate Data
Generate synthetic data for classification using `generate_data()`.

```python
X, y = generate_data()
```

### Step 2: Run Logistic Regression
Fit a logistic regression model using gradient descent:

```python
cost_list, optimal_parameters = logistic_regression(X, y)
```

### Step 3: Plot Decision Boundary
Visualize the decision boundary of the model:

```python
plot_decision_boundary(X, y, optimal_parameters)
```

### Step 4: Perform Assumption Testing
Check the assumptions of logistic regression:

```python
assumption_testing(X, y, optimal_parameters)
```

## Assumptions Tested

- **Linearity of Independent Variables and Log Odds**: Assumes that the relationship between independent variables and the log odds of the dependent variable is linear.
- **No Strongly Influential Outliers**: Assumes that there are no data points that disproportionately influence the model.
- **Absence of Multicollinearity**: Assumes that the independent variables are not highly correlated with each other (measured using VIF).
- **Sufficiently Large Sample Size**: Assumes that the dataset is large enough to produce reliable estimates.

## Example Output

- **Model Summary**: Displays key statistics about the model fit.
- **Cook's Distance**: Identifies influential data points.
- **Variance Inflation Factors (VIF)**: Checks for multicollinearity.
- **Decision Boundary Plot**: Visualizes the boundary separating the two classes.

## Notes

- The Logistic Regression implementation is based on standard gradient descent optimization.
- The assumption testing portion uses statistical methods to ensure the suitability of the logistic regression model for the data.
