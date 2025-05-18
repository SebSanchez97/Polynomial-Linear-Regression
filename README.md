# Project Description: Custom Polynomial Regression Model (Second-Order and Multi-Order)

## Data Source:
This project generates a synthetic dataset for polynomial regression:

Number of Samples (N): 100 samples.

Feature (X): Random values generated in the range [-3, 3].

Target (y): Generated using a quadratic equation:

𝑦
=
2
+
𝑥
+
0.5
𝑥
2
+
𝜖
y=2+x+0.5x 
2
 +ϵ
where:

𝑥
x is the generated input feature.

𝜖
ϵ is a random noise term added to introduce variability.

Machine Learning Model Architecture:
This project implements two types of custom Polynomial Regression models:

1. Second-Order Polynomial Regression Model:
The model is defined in the SecondOrderPolynomialRegressionModel class in polyRegressionModel.py.

Model Architecture:

The input feature (X) is transformed to include a second-order term (
𝑋
2
X 
2
 ).

A design matrix is created, including the first-order term, second-order term, and a bias term.

The model is trained using the Ordinary Least Squares (OLS) method to calculate the weights.

## Training Process:

The model fits the training data by minimizing the Mean Squared Error (MSE).

The weights for the polynomial terms are calculated analytically.

## Visualization:

The trained quadratic model is visualized using a scatter plot of the data and the regression curve.

2. Multi-Order Polynomial Regression Model:
The model is defined in the MultiOrderPolynomialRegressionModel class in polyRegressionModel.py.

## Model Architecture:

The input feature (X) is transformed into a higher-order polynomial of the specified degree (poly_order).

A design matrix is created with polynomial terms up to the specified degree.

The model is trained using the Ordinary Least Squares (OLS) method.

## Training Process:

The model fits the data by minimizing the Mean Squared Error (MSE).

The weights for the polynomial terms are calculated analytically.

## Visualization:

The trained polynomial model is visualized using a scatter plot of the data and the regression curve.

## Usage:
The main file (main.py) demonstrates how the models are used:

A second-order polynomial regression model is trained and visualized.

A multi-order polynomial regression model is trained and visualized for a specified polynomial order (default is 20).