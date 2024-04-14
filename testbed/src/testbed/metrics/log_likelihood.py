from typing import Dict
from typing import Optional

import numpy as np
from jaxtyping import Float
from numpy import ndarray
from sklearn.neighbors import KernelDensity

from testbed.metrics.base_metric import Metric
from testbed.models.base_model import ProbabilisticModel


class LogLikelihoodFromSamplesMetric(Metric):
    """
    Computes the log likelihood of a model's predictive distribution given empirical samples of the model.

    Parameters
    ----------
    n_samples : int, optional
        The number of samples to draw from the model's predictive distribution.
    bandwidth : float, optional
        The bandwidth of the kernel density estimator used to fit the samples.
        If None, the bandwidth is estimated using cross-validation.
    """

    def __init__(self, n_samples: int = 100, bandwidth: Optional[float] = 1.0) -> None:

        self.n_samples = n_samples
        self.bandwidth = bandwidth

    def compute(
        self,
        model: ProbabilisticModel,
        X_test: Float[ndarray, "batch n_features"],
        y_test: Float[ndarray, "batch y_dim"],
    ) -> Dict[str, float]:
        """
        Compute the log likelihood of the predictive distribution.

        Parameters
        ----------
        model : ProbabilisticModel
            The model to evaluate.
        X_test : ndarray of shape (batch, n_features)
            The input data.
        y_test : ndarray of shape (batch, y_dim)
            The true output values.

        Returns
        -------
        log_likelihood : dict
            A single scalar which quantifies the log likelihood of the predictive distribution from empirical samples.
        """

        y_samples: Float[ndarray, "n_samples batch y_dim"] = model.sample(
            X=X_test, n_samples=self.n_samples
        )

        def fit_and_evaluate_kde(y_train: Float[ndarray, "n_samples y_dim"], y_test):
            if self.bandwidth is not None:
                kde = KernelDensity(bandwidth=self.bandwidth)
            else:
                # fit a kernel density estimator to the samples using cross-validation for the bandwidth
                kde = KernelDensity()
                from sklearn.model_selection import GridSearchCV

                grid = GridSearchCV(kde, {"bandwidth": np.logspace(-1, 2, 10)}, cv=5)
                grid.fit(y_train)
                kde = grid.best_estimator_

            kde.fit(y_train)

            return kde.score_samples(y_test).item()

        n_samples, batch, y_dim = y_samples.shape

        assert batch == X_test.shape[0]
        assert y_dim == y_test.shape[1]
        assert n_samples == self.n_samples

        nll = 0
        for i in range(batch):
            y_train_xi = y_samples[:, i, :]
            y_test_xi = y_test[i, :]
            nll -= fit_and_evaluate_kde(y_train_xi, [y_test_xi])

        return {
            "nll": nll,
        }