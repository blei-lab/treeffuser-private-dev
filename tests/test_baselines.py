import numpy as np
import pytest
from scipy.stats import ks_2samp

from treeffuser.baselines.nnffuser import NNffuser

from .utils import gaussian_mixture_pdf
from .utils import train_test_split


@pytest.mark.parametrize("card_like", [False])
def test_baselines_bimodal_linear_regression(card_like):
    """
    We do a very simple sanity check that for a very simple model with not enough data the
    samples from the model should be statistically indistinguishable from the data.
    """
    n = 1000
    n_samples = 1
    rng = np.random.default_rng(seed=0)

    X_1 = rng.uniform(size=(n, 1))
    y_1 = X_1 + rng.normal(size=(n, 1)) * 0.05 * (X_1 + 1) ** 2

    X_2 = rng.uniform(size=(n, 1))
    y_2 = -X_2 + rng.normal(size=(n, 1)) * 0.05 * (X_2 + 1) ** 2

    X = np.concatenate([X_1, X_2], axis=0)
    y = np.concatenate([y_1, y_2], axis=0)

    # Shuffle and split the data
    idx = rng.permutation(2 * n)
    X = X[idx]
    y = y[idx]

    X_train = X[:n]
    y_train = y[:n]

    X_test = X[-100:]
    y_test = y[-100:]

    model = NNffuser(
        verbose=1,
        n_repeats=200,
        card_like=card_like,
        sde_name="vesde",
        batch_size=256,
        n_layers=3,
        hidden_size=100,
        learning_rate=0.005,
        ema_decay=0.0,
        early_stopping_rounds=100,
        eval_freq=100,
        max_evals=1000,
        seed=0,
    )
    model.fit(X_train, y_train)
    y_samples = model.sample(X_test, n_samples=n_samples, n_parallel=50, n_steps=30, seed=0)

    y_samples = y_samples.flatten()
    y_test = y_test.flatten()

    # Check that the samples are statistically indistinguishable from the data
    result = ks_2samp(y_samples, y_test)
    assert result.pvalue > 0.05, f"p-value: {result.pvalue}"


def test_sample_based_nll_gaussian_mixture():
    """
    The data are generated from a Gaussian mixture model with conditional density:
    p(y_i | x_i) = .5 * N(x_i, x_i ** 2) + (1 - .5) * N(-x_i, x_i ** 2)
    """
    n = 10**3
    rng = np.random.default_rng(seed=0)

    x = rng.uniform(low=1, high=2, size=(n, 1))
    sign = 2 * rng.binomial(n=1, p=0.5, size=(n, 1)) - 1
    y = rng.normal(loc=sign * x, scale=abs(x), size=(n, 1))

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.04, random_state=42)

    model = NNffuser(
        verbose=1,
        n_repeats=100,
        max_evals=100000,
        eval_freq=100,
        hidden_size=100,
        n_layers=3,
        sde_name="vesde",
        learning_rate=0.005,
        ema_decay=0.0,
        early_stopping_rounds=20,
        seed=0,
    )
    model.fit(x_train, y_train)

    nll_treeffuser = model.compute_nll(x_test, y_test, ode=False, n_samples=10**2, bandwidth=1)
    nll_true = -(
        gaussian_mixture_pdf(
            y_test, x_test, np.abs(x_test), -x_test, np.abs(x_test), 0.5, log=True
        )
        .sum()
        .item()
    )

    relative_error = np.abs(nll_treeffuser / nll_true - 1)
    assert relative_error < 0.05, f"relative error: {relative_error}"


def test_categorical():
    """Basic test for categorical variable support."""
    n = 10**3
    rng = np.random.default_rng(seed=0)

    X_noncat = rng.uniform(low=1, high=2, size=(n, 1))
    X_cat = rng.choice(1, size=(n, 1))
    X = np.concatenate([X_noncat, X_cat], axis=1)

    y = rng.normal(loc=X_noncat + 2 * X_cat, scale=1, size=(n, 1))

    for cat_idx in [None, [1]]:
        model = NNffuser()
        model.fit(X=X, y=y, cat_idx=cat_idx)
