from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.unique(y)
        self.mu_ = np.ndarray((len(self.classes_), len(X[1])))
        self.cov_ = np.ndarray((len(X[1]), len(X[1])))
        self.pi_ = np.ndarray((len(self.classes_),))
        m = len(X)
        for class_k in range(len(self.classes_)):
            self.mu_[class_k] = np.array(X[y == self.classes_[class_k]].mean(axis=0))
            self.pi_[class_k] = np.count_nonzero(y == self.classes_[class_k]) / m
            class_cov = X[y == self.classes_[class_k]] - self.mu_[class_k,:]
            self.cov_ += class_cov.T @ class_cov
        self.cov_ = self.cov_ / (m - len(self.classes_))
        self._cov_inv = np.linalg.inv(self.cov_)

    def _predict(self, X: np.ndarray) -> np.ndarray:
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
        class_index = np.argmax(self.likelihood(X), axis=1)
        return np.take(self.classes_, class_index)

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        likelihood = np.ndarray((len(X), len(self.classes_)))

        for m in range(len(X)):
            cur_vec = np.ndarray((len(self.classes_),))
            for k in range(len(self.classes_)):
                a_k = self._cov_inv @ self.mu_[k]
                b_k = np.log(self.pi_[k]) - 0.5 * (self.mu_[k] @ self._cov_inv @ self.mu_[k])
                cur_vec[k] = np.transpose(a_k) @ X[m] + b_k
            likelihood[m] = cur_vec
        return likelihood

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
        from ...metrics import misclassification_error
        return misclassification_error(y, self.predict(X))
