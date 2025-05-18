import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

class SecondOrderPolynomialRegressionModel:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.X_design = None
        self.weights = None

    def train(self):
        # Add second-order polynomial term to the end of the design matrix
        self.X_design = np.concatenate((self.X, (self.X * self.X)), axis=1)

        # Add bias terms to design matrix
        bias_vector = np.ones((self.X.shape[0], 1), dtype=np.int32)
        self.X_design = np.concatenate((bias_vector, self.X_design), axis=1)

        # Calculates and returns optimal weights through use of the normal equations
        self.weights = np.linalg.inv(self.X_design.T @ self.X_design) @ self.X_design.T @ self.y

    def plot_second_order_quadratic_regression_model(self):
        plt.scatter(self.X, self.y)
        plt.xlabel("x")
        plt.ylabel("y")

        x_plot = np.linspace(np.min(self.X), np.max(self.X), self.X.shape[0])
        x_design = np.reshape(x_plot, (x_plot.shape[0], 1))
        
        # Add second-order polynomial term to the end of the design matrix
        x_design = np.concatenate((x_design, (x_design * x_design)), axis=1)

        # Add bias terms to data matrix
        bias_vector = np.ones((x_design.shape[0], 1), dtype=np.int32)
        x_design = np.concatenate((bias_vector, x_design), axis=1)

        prediction = x_design @ self.weights.T

        plt.plot(x_plot, prediction, color="red")
        plt.title("Second-Order Polynomial regression model")
        plt.show()

class MultiOrderPolynomialRegressionModel:
    def __init__(self, X, y, poly_order):
        self.order = poly_order
        self.X = X
        self.y = y
        self.weights = None

    def train_polynomial_model(self):
        '''
        Calculate the least-squares weights for polynomial regression using the normal equations
        '''
        polynomial_features = PolynomialFeatures(degree=self.order, include_bias=True)

        X_poly = polynomial_features.fit_transform(self.X)

        self.weights = np.linalg.inv(X_poly.T @ X_poly) @ X_poly.T @ self.y
    
    def plot_model(self):
        '''
        Plot the model's prediction
        '''
        X_new = np.linspace(np.min(self.X), np.max(self.X), self.X.shape[0])

        X_new = np.reshape(X_new, (X_new.shape[0], 1))

        polynomial_features = PolynomialFeatures(degree=self.order, include_bias=True)

        X_new_poly = polynomial_features.fit_transform(X_new)

        y_hat = X_new_poly @ self.weights

        plt.scatter(self.X, self.y)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.plot(X_new, y_hat, "red")
        plt.show()