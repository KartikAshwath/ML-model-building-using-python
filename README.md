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


# Multiple Linear Regression Model for Delivery Time Prediction

This notebook demonstrates the process of building and evaluating a multiple linear regression model to predict delivery time based on the number of products and distance.

## Dataset

The dataset used in this notebook is `delivery.csv`. It contains the following columns:

- `n.prod`: Number of products to be delivered.
- `distance`: Distance to be covered for delivery.
- `delTime`: Delivery time (target variable).

## Notebook Structure

1.  **Import Libraries**: Imports necessary libraries like pandas, numpy, matplotlib, and seaborn.
2.  **Load Data**: Reads the `delivery.csv` file into a pandas DataFrame.
3.  **Feature Engineering**:
    - Provides basic data information (`.info()`) and descriptive statistics (`.describe()`).
    - Visualizes the relationships between variables using a pairplot.
4.  **Building Multiple Linear Model**:
    - Instantiates and fits a `LinearRegression` model from scikit-learn using `n.prod` and `distance` as predictors and `delTime` as the target.
    - Prints the intercept and coefficients of the model.
5.  **Visualizing the Model**:
    - Creates a 3D scatter plot of the data.
    - Plots the regression plane on the 3D scatter plot to visualize the model's fit.
6.  **Validating the Model**:
    - Calculates and prints the R-squared value of the multiple linear regression model.
7.  **Comparison between model r-squared based on number of predictors**:
    - Compares the R-squared of the multiple linear regression model with a simple linear regression model using only `n.prod`.
8.  **Computation of adjusted R-Squared**:
    - Calculates and prints the adjusted R-squared value for the multiple linear regression model.
9.  **Feature Engineering in Detail for Multiple Linear Regression**:
    - Calculates and prints the correlation between the predictors (`n.prod` and `distance`).
    - Calculates and prints the Variance Inflation Factor (VIF) for each predictor to assess multicollinearity.

## How to Use

1.  Ensure you have the `delivery.csv` file in the same directory as the notebook or provide the correct path.
2.  Run the cells sequentially to execute the code and see the results.

## Requirements

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- statsmodels
"""

```bash
pip install pandas numpy matplotlib scikit-learn

