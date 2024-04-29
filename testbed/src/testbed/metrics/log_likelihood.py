from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np
from jaxtyping import Float
from numpy import ndarray
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from tqdm import tqdm

from testbed.metrics.base_metric import Metric
from testbed.models.base_model import ProbabilisticModel

# Corresponds to event with probability of 10^-6
INT_LIKELIHOOD_CUTOFF = -6


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
    is_int : bool, optional
        Whether the samples are meant to be integers. If True, the
        likelihood is estimated via simple binning to the neares integer.
    verbose : bool, optional
        Whether to print progress information.
    """

    def __init__(
        self,
        n_samples: int = 50,
        bandwidth: Optional[Union[str, float]] = "scott",
        is_int=False,
        verbose=False,
    ) -> None:

        self.n_samples = n_samples
        self.bandwidth = bandwidth
        self.is_int = is_int
        self.verbose = verbose

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
        # use seaborn to plot stuff

        y_samples: Float[ndarray, "n_samples batch y_dim"] = model.sample(
            X=X_test, n_samples=self.n_samples
        )
        n_samples, batch, y_dim = y_samples.shape

        assert batch == X_test.shape[0], f"batch={batch} != X_test.shape[0]={X_test.shape[0]}"
        assert y_dim == y_test.shape[1], f"y_dim={y_dim} != y_test.shape[1]={y_test.shape[1]}"
        assert n_samples == self.n_samples

        nll = 0
        for i in tqdm(range(batch), disable=not self.verbose):
            y_train_xi = y_samples[:, i, :]
            y_test_xi = y_test[i, :]

            if self.is_int:
                nll -= fit_and_evaluate_int_prob(y_train_xi, [y_test_xi])
            else:
                nll -= fit_and_evaluate_kde(y_train_xi, [y_test_xi], bandwidth=None)

        return {
            "nll": nll,
        }


def fit_and_evaluate_int_prob(y_train: Float[ndarray, "n_samples y_dim"], y_test: List[float]):
    """
    Used only when y_train is meant to represent integer values.

    We fit an empirical distribution to the training data using simple counting.
    We leave one count for unseen values.
    """
    # Currently only works with 1D y_train
    assert y_train.shape[1] == 1

    y_test_int = np.round(y_test).flatten()
    y_train_int = np.round(y_train).flatten()

    unique, counts = np.unique(y_train_int, return_counts=True)
    total_counts = np.sum(counts) + 1  # one extra count for unseen values

    counts_dict = dict(zip(unique, counts))
    probs_test = [counts_dict.get(y, 1) / total_counts for y in y_test_int]
    log_probs_test = np.log(probs_test)
    return np.sum(log_probs_test)


def fit_and_evaluate_kde(y_train: Float[ndarray, "n_samples y_dim"], y_test, bandwidth=None):
    if bandwidth is not None:
        kde = KernelDensity(bandwidth=bandwidth)
    else:
        # fit a kernel density estimator to the samples using cross-validation for the bandwidth
        kde = KernelDensity()
        std = np.std(y_train)
        log_std = np.log10(std)
        grid = GridSearchCV(
            kde,
            {"bandwidth": np.logspace(log_std - 3, log_std + 3, 10, base=10)},
            cv=3,
        )
        grid.fit(y_train)
        kde = grid.best_estimator_

    # plot the kde
    kde.fit(y_train)

    score = kde.score_samples(y_test)[0]
    return score
