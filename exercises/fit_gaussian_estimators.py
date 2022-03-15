from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"
MU_UNI = 10
SIGMA_UNI = 1


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    X = np.random.normal(MU_UNI, SIGMA_UNI, 1000)
    data = UnivariateGaussian()
    data.fit(X)
    print((data.mu_, data.var_))

    # Question 2 - Empirically showing sample mean is consistent
    slicing_samples = np.arange(10, 1010, 10)
    calculate_y_values = np.vectorize(lambda slice: np.abs(data.fit(X[:slice]).mu_-MU_UNI))
    y_values = calculate_y_values(slicing_samples)

    fig1 = go.Figure(
        data=[go.Bar(x=slicing_samples, y=y_values)])

    fig1.update_layout(title="distance between the estimated and true value of the expectation,"
                             "as a function of the sample size",
                      xaxis_title = "sample size",
                      yaxis_title = "distance between the estimated and true value of the expectation")
    fig1.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdf_values = data.pdf(X)
    fig2 = go.Figure(data=go.Scatter(x=X, y=pdf_values, mode='markers'))

    fig2.update_layout(title="PDF",
                       xaxis_title="sample value",
                       yaxis_title="PDF value")

    fig2.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    sigma = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    data = MultivariateGaussian()
    data.fit(np.random.multivariate_normal(mu, sigma, 1000))
    print(data.mu_, data.cov_)

    # Question 5 - Likelihood evaluation
    raise NotImplementedError()

    # Question 6 - Maximum likelihood
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
