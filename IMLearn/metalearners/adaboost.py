import numpy as np
from IMLearn import BaseEstimator
from typing import Callable, NoReturn
from IMLearn.metrics import misclassification_error


class AdaBoost(BaseEstimator):
    """
    AdaBoost class for boosting a specified weak learner

    Attributes
    ----------
    self.wl_: Callable[[], BaseEstimator]
        Callable for obtaining an instance of type BaseEstimator

    self.iterations_: int
        Number of boosting iterations to perform

    self.models_: List[BaseEstimator]
        List of fitted estimators, fitted along the boosting iterations
    """

    def __init__(self, wl: Callable[[], BaseEstimator], iterations: int):
        """
        Instantiate an AdaBoost class over the specified base estimator

        Parameters
        ----------
        wl: Callable[[], BaseEstimator]
            Callable for obtaining an instance of type BaseEstimator

        iterations: int
            Number of boosting iterations to perform
        """
        super().__init__()
        self.wl_ = wl
        self.iterations_ = iterations
        self.models_, self.weights_, self.D_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an AdaBoost classifier over given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        m = len(X)
        self.D_ = np.ones((m,)) / m
        self.models_ = np.ndarray(shape=(self.iterations_,), dtype=BaseEstimator)
        self.weights_ = np.ndarray(shape=(self.iterations_,))
        for t in range(self.iterations_):
            # combine_sample = np.c_[X, y]
            # if t != 0:
            #     index_list = np.random.choice(m, size=(m,), p=self.D_)
            #     combine_sample = combine_sample[index_list]
            # new_X = combine_sample[:, :-1]
            # new_y = combine_sample[:, -1]
            ht = self.wl_().fit(X, y * self.D_)
            y_pred = ht.predict(X)
            epsilon_t = np.sum(self.D_[y_pred != y])
            self.weights_[t] = 0.5 * np.log((1 / epsilon_t) - 1)
            self.D_ *= np.exp(-(y * self.weights_[t] * y_pred))
            self.D_ = self.D_ / (np.sum(self.D_))
            self.models_[t] = ht

    def _predict(self, X):
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        models_response = np.ndarray(shape=(len(self.models_), len(X)))
        for i in range(self.models_):
            models_response[i] = self.models_[i].predict(X)
        responses = np.sum((self.weights_ * models_response), axis=0)
        return np.sign(responses)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return misclassification_error(y, self.predict(X))

    def partial_predict(self, X: np.ndarray, T: int) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimators

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        models_response = np.zeros(shape=(len(X),))
        for i in range(T):
            models_response += (self.models_[i].predict(X) * self.weights_[i])
        return np.sign(models_response)

    def partial_loss(self, X: np.ndarray, y: np.ndarray, T: int) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return misclassification_error(y, self.partial_predict(X, T))
