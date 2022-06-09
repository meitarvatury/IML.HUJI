from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from IMLearn import BaseEstimator

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """

    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    def f(x):
        return (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)

    X = np.linspace(-1.2, 2, n_samples)
    epsilon = np.random.normal(0, noise, n_samples)
    y = np.vectorize(f)(X) + epsilon
    train_X, train_y, test_X, test_y = split_train_test(X=pd.DataFrame(X), y=pd.Series(y), train_proportion=2 / 3)
    train_X = train_X.to_numpy().reshape(train_X.shape[0], )
    train_y = train_y.to_numpy().reshape(train_X.shape[0], )
    test_X = test_X.to_numpy().reshape(test_X.shape[0], )
    test_y = test_y.to_numpy().reshape(test_X.shape[0], )

    fig1 = go.Figure()

    fig1.add_trace(
        go.Scatter(x=X, y=f(X), mode="markers", name='noiseless data set'))

    fig1.add_trace(
        go.Scatter(x=train_X, y=train_y,
                   mode="markers", name='train set'))

    fig1.add_trace(
        go.Scatter(x=test_X, y=test_y,
                   mode="markers", name='test set'))

    fig1.update_layout(title="the true (noiseless) data set and the two(train and test) sets",
                       xaxis_title="x values",
                       yaxis_title="y values")

    fig1.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    k_values = np.arange(11)
    train_scores = []
    validation_scores = []
    for k in k_values:
        train_score, validation_score = cross_validate(PolynomialFitting(k), train_X, train_y, mean_square_error)
        train_scores.append(train_score)
        validation_scores.append(validation_score)

    fig2 = go.Figure()

    fig2.add_trace(
        go.Scatter(x=k_values, y=train_scores,
                   mode="markers+lines", name='train errors'))

    fig2.add_trace(
        go.Scatter(x=k_values, y=validation_scores,
                   mode="markers+lines", name='validation errors'))

    fig2.update_layout(title="the train and validation errors as a function of the degree of the polynomial model",
                       xaxis_title="degree of the polynomial model",
                       yaxis_title="the errors values from cross validation")

    fig2.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    best_k = int(np.argmin(validation_scores))
    fitted_model = PolynomialFitting(best_k).fit(train_X, train_y)
    test_error = mean_square_error(test_y, fitted_model.predict(test_X))
    print("Best k value: " + str(best_k))
    print("validation error for best k fitting: " + str(np.round(validation_scores[best_k])))
    print("test error for best k fitting: " + str(np.round(test_error, 2)))


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    X_train, y_train = X[:n_samples, :], y[:n_samples]
    X_test, y_test = X[n_samples:, :], y[n_samples:]

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lambdas = np.linspace(0.001, 2, num=n_evaluations)
    train_scores_ridge = []
    validation_scores_ridge = []
    train_scores_lasso = []
    validation_scores_lasso = []
    for i in range(n_evaluations):
        train_score_ridge, validation_score_ridge = cross_validate(RidgeRegression(lambdas[i]), X_train, y_train,
                                                                   mean_square_error)
        train_scores_ridge.append(train_score_ridge)
        validation_scores_ridge.append(validation_score_ridge)
        train_score_lasso, validation_score_lasso = cross_validate(Lasso(alpha=lambdas[i]), X_train, y_train,
                                                                   mean_square_error)
        train_scores_lasso.append(train_score_lasso)
        validation_scores_lasso.append(validation_score_lasso)

    fig3 = make_subplots(rows=1, cols=2, subplot_titles=["Lasso regularization", "ridge regularization"],
                         horizontal_spacing=0.01, vertical_spacing=.03)

    fig3.add_traces([go.Scatter(x=lambdas, y=train_scores_lasso, mode="markers", name='lasso train errors'),
                     go.Scatter(x=lambdas, y=validation_scores_lasso, mode="markers",
                                name='lasso validation errors')],
                    rows=1, cols=1)

    fig3.add_traces([go.Scatter(x=lambdas, y=train_scores_ridge, mode="markers", name='ridge train errors'),
                     go.Scatter(x=lambdas, y=validation_scores_ridge, mode="markers",
                                name='ridge validation errors')],
                    rows=1, cols=2)

    fig3.update_layout(title="validation and train errors using ridge and lasso regularization", margin=dict(t=100))

    fig3.show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    min_val_ridge = np.argmin(validation_scores_ridge)
    min_val_lasso = np.argmin(validation_scores_lasso)

    ridge_model = RidgeRegression(lambdas[min_val_ridge])
    ridge_model.fit(X_train, y_train)
    min_test_error_ridge = mean_square_error(ridge_model.predict(X_test), y_test)

    lasso_model = Lasso(alpha=lambdas[min_val_lasso])
    lasso_model.fit(X_train, y_train)
    min_test_error_lasso = mean_square_error(y_test, lasso_model.predict(X_test))

    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    min_test_error_linear = mean_square_error(linear_model.predict(X_test), y_test)

    print("ridge loss: " + str(min_test_error_ridge))
    print("lasso loss: " + str(min_test_error_lasso))
    print("linear loss: " + str(min_test_error_linear))



if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(n_samples=100, noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)
    select_regularization_parameter()
