from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"
MU_UNI = 10
SIGMA_UNI = 1
SAMPLE_SIZE = 1000


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    X = np.random.normal(MU_UNI, SIGMA_UNI, SAMPLE_SIZE)
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
                      xaxis_title ="sample size",
                      yaxis_title ="distance between the estimated and true value of the expectation")
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
    X = np.random.multivariate_normal(mu, sigma, SAMPLE_SIZE)
    data.fit(X)
    print(data.mu_)
    print(data.cov_)

    # Question 5 - Likelihood evaluation
    lin_range = np.linspace(-10, 10, 200)
    likelihood_values = []
    likelihood_args_dict = {}
    for f1 in lin_range:
        res = []
        for f3 in lin_range:
            log_likelihood = data.log_likelihood(np.array([f1, 0, f3, 0]), sigma, X)
            res.append((data.log_likelihood(np.array([f1, 0, f3, 0]), sigma, X)))
            likelihood_args_dict[log_likelihood] = (f1, f3)
        likelihood_values.append(res)

    fig3 = go.Figure(
        data=[go.Heatmap(x=lin_range, y=lin_range, z=likelihood_values)])

    fig3.update_layout(title="Likelihood evaluation as a function of f1,f3 values when mu = [f1,0,f3,0]",
                       xaxis_title="f3 values",
                       yaxis_title="f1 values")
    fig3.show()


    # Question 6 - Maximum likelihood
    print(np.round(likelihood_args_dict[np.max(likelihood_values)], 3))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
