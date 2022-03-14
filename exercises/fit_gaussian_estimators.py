from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    data = UnivariateGaussian()
    data.fit(np.random.normal(10, 1, 1000))
    print((data.mu_, data.var_))

    # Question 2 - Empirically showing sample mean is consistent
    #fig = go.Figure(data=[go.Histogram(x=data)])
    #fig.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    raise NotImplementedError()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    sigma = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    data = MultivariateGaussian()
    data.fit(np.random.multivariate_normal(mu, sigma, 1000))
    print((data.mu_, data.cov_))

    # Question 5 - Likelihood evaluation
    raise NotImplementedError()

    # Question 6 - Maximum likelihood
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
