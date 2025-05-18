import numpy as np
import polyRegressionModel

# Create synthetic data
N = 100
x = 6*np.random.rand(N) - 3
y = 2 + x + 0.5 * x**2 + np.random.randn(N)
X = x.reshape(N, 1)

poly_model = polyRegressionModel.SecondOrderPolynomialRegressionModel(X=X, y=y)
poly_model.train()
poly_model.plot_second_order_quadratic_regression_model()

polynomial_order = 20
multi_poly_model = polyRegressionModel.MultiOrderPolynomialRegressionModel(X=X, y=y, poly_order=polynomial_order)
multi_poly_model.train_polynomial_model()

multi_poly_model.plot_model()
