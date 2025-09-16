# ML-model-building-using-python
# Simple Linear Regression: Manual and Scikit-Learn Implementation

This notebook demonstrates how to build and evaluate a simple linear regression model using both a manual approach and the scikit-learn library in Python. The goal is to predict the time taken to repair a computer based on the number of units to be repaired.

## Dataset

The dataset used in this notebook is `computers.csv`. It contains two columns:

- `Units`: The number of units to be repaired (predictor variable).
- `Minutes`: The time taken to repair the units (target variable).

## Agenda

The notebook covers the following topics:

- Building a simple linear regression model manually.
- Building a simple linear regression model using the Scikit-Learn library.

## Steps Involved

The process of building and evaluating the models includes the following steps:

1.  **Read Data**: Load the `computers.csv` dataset into a pandas DataFrame.
2.  **Feature Engineering**: Explore the data and visualize the relationship between the predictor and target variables.
3.  **Create Train and Test Sets**: Although not explicitly done with `train_test_split` in this specific manual example, the concept of using data to build and test the model is demonstrated.
4.  **Build Model Manually**:
    -   Understand the simple linear regression equation: y = mx + c.
    -   Calculate the slope (m) and intercept (c) using formulas derived from differential calculus to minimize the sum of squared errors (SSE).
    -   The formulas used are:
        -   `m = (sum(x*y) - n*mean(x)*mean(y)) / (sum(x**2) - n*(mean(x)**2))`
        -   `c = mean(y) - (m * mean(x))`
        where x is the predictor values, y is the actual target values, and n is the sample size.
5.  **Build Model using Scikit-Learn**:
    -   Import the `LinearRegression` class from `sklearn.linear_model`.
    -   Create a `LinearRegression` model instance.
    -   Fit the model to the data using the `fit()` method.
    -   Obtain the intercept and coefficient from the fitted model.
6.  **Test the Model**:
    -   Calculate the predicted values using the built models.
    -   Calculate the error (difference between actual and predicted values).
7.  **Visualize**:
    -   Plot the actual data points and the regression lines from the different models (including the best-fit model).
8.  **Evaluate the Model**:
    -   Calculate the Sum of Squared Errors (SSE) for the models to assess their performance.
    -   Calculate the Total Sum of Squares (SST) and Sum of Squares Regression (SSR).
    -   Calculate the Coefficient of Determination (R-squared) using the formulas `Rsq = SSR/SST` or the `score()` method of the scikit-learn model. R-squared indicates the proportion of the variance in the dependent variable that is predictable from the independent variable.

## Running the Notebook

To run this notebook, you will need to have the following libraries installed:

-   pandas
-   numpy
-   matplotlib
-   scikit-learn

You can install them using pip:

```bash
pip install pandas numpy matplotlib scikit-learn
