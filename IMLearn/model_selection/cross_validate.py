from __future__ import annotations
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    x_split = np.array_split(X, cv)
    y_split = np.array_split(y, cv)
    train_scores = np.ones(cv)
    validation_scores = np.ones(cv)
    for i in range(cv):
        train_data, labels = np.concatenate(np.delete(x_split, i, 0), axis=0), \
                      np.concatenate(np.delete(y_split, i, 0), axis=0)
        fitted_estimator = estimator.fit(train_data, labels)
        train_scores[i] = scoring(labels, fitted_estimator.predict(train_data))
        validation_scores[i] = scoring(y_split[i], fitted_estimator.predict(x_split[i]))
    return np.average(train_scores), np.average(validation_scores)
