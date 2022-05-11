import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
from IMLearn.metrics.loss_functions import accuracy
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaboost = AdaBoost(wl=DecisionStump, iterations=n_learners).fit(train_X, train_y)
    error_x = np.arange(1, n_learners + 1)
    error_y_test = np.ones(shape=(n_learners,))
    error_y_train = np.ones(shape=(n_learners,))
    for i in range(n_learners):
        error_y_test[i] = adaboost.partial_loss(test_X, test_y, i + 1)
        error_y_train[i] = adaboost.partial_loss(train_X, train_y, i + 1)

    fig1 = go.Figure()

    fig1.add_trace(go.Scatter(x=error_x, y=error_y_test, mode='markers+lines', name='test set'))

    fig1.add_trace(go.Scatter(x=error_x, y=error_y_train, mode="markers+lines", name='train set'))

    fig1.update_layout(title="loss values as a function of the number of fitted learners",
                       xaxis_title="number of fitted learners",
                       yaxis_title="loss values")

    fig1.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    symbols = np.array(["circle", "x"])
    fig2 = make_subplots(rows=2, cols=2, subplot_titles=["5 iterations", "50 iterations", "100 iterations",
                                                         "250 iterations"],
                         horizontal_spacing=0.01, vertical_spacing=.03)

    def decision_surface(predict, xrange, yrange, T, density=120, dotted=False, colorscale=custom, showscale=True):
        xrange, yrange = np.linspace(*xrange, density), np.linspace(*yrange, density)
        xx, yy = np.meshgrid(xrange, yrange)
        pred = predict(np.c_[xx.ravel(), yy.ravel()], T)

        if dotted:
            return go.Scatter(x=xx.ravel(), y=yy.ravel(), opacity=1, mode="markers",
                              marker=dict(color=pred, size=1, colorscale=colorscale, reversescale=False),
                              hoverinfo="skip", showlegend=False)
        return go.Contour(x=xrange, y=yrange, z=pred.reshape(xx.shape), colorscale=colorscale, reversescale=False,
                          opacity=.7, connectgaps=True, hoverinfo="skip", showlegend=False, showscale=showscale)

    for i, t in enumerate(T):
        fig2.add_traces([decision_surface(adaboost.partial_predict, lims[0], lims[1], T=t, showscale=False),
                         go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                    marker=dict(color=test_y,
                                                symbol=symbols[np.where(test_y < 0, 0, test_y).astype(int)]
                                                , colorscale=[custom[0], custom[-1]],
                                                line=dict(color="black", width=1)))],
                        rows=(i // 2) + 1, cols=(i % 2) + 1)

    fig2.update_layout(title="decision boundary obtained after different number of iterations", margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)

    fig2.show()

    # Question 3: Decision surface of best performing ensemble
    min_error_t = np.argmin(error_y_test)
    accuracy_min_t = accuracy(test_y, adaboost.partial_predict(test_X, min_error_t))

    fig3 = go.Figure([decision_surface(adaboost.partial_predict, lims[0], lims[1], T=min_error_t, showscale=False),
                      go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                 marker=dict(color=test_y, symbol=symbols[np.where(test_y < 0, 0, test_y).astype(int)]
                                             , colorscale=[custom[0], custom[-1]], line=dict(color="black", width=1)))])

    fig3.update_layout(title="decision surface with ensemble size "+str(min_error_t)+" and with accuracy "+str(accuracy_min_t),
                       margin=dict(t=100)).update_xaxes(visible=False).update_yaxes(visible=False)

    fig3.show()

    # Question 4: Decision surface with weighted samples
    D = (adaboost.D_ / np.max(adaboost.D_)) * 5
    fig4 = go.Figure([decision_surface(adaboost.partial_predict, lims[0], lims[1], T=250, showscale=False),
                      go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                                 marker=dict(color=train_y,
                                             symbol=symbols[np.where(train_y < 0, 0, train_y).astype(int)], size=D
                                             , colorscale=[custom[1], custom[2]], line=dict(color="black", width=1)))])

    fig4.update_layout(title="the training set with a point size proportional to it’s weight after running adaboost",
                       margin=dict(t=100)).update_xaxes(visible=False).update_yaxes(visible=False)

    fig4.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(noise=0, n_learners=250)
    fit_and_evaluate_adaboost(noise=0.4, n_learners=250)
