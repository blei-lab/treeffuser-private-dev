"""
This file should contain a general abstraction of the score models and
should function as a wrapper for different models we might want to use.
"""

import abc
from typing import List
from typing import Optional
from typing import Callable

import lightgbm as lgb
import numpy as np
from jaxtyping import Float
from jaxtyping import Int
from sklearn.model_selection import train_test_split

from treeffuser.sde import DiffusionSDE
from treeffuser._score_models._base import Score

import torch as t
import torch.nn as nn
from torch.utils.data import DataLoader

###################################################
# Helper functions and classes
###################################################

class _MLPModule(nn.Module):
    def __init__(self, num_layers: int, hidden_size: int, input_size: int, output_size: int):
        """
        Simple MLP model with ReLU activation functions.
        """
        super(_MLPModule, self).__init__()
        layers = []

        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_size, output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x: Float[t.Tensor, "batch x_dim"]) -> Float[t.Tensor, "batch y_dim"]:
        return self.model(x)

def _evaluate_model(
    model : nn.Module,
    data_loader : DataLoader,
    criterion : Callable
) -> float:
    model.eval()
    total_loss = 0.0
    with t.no_grad():
        for data, target in data_loader:
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()

    return total_loss / len(data_loader)

def _train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: Callable,
    optimizer: t.optim.Optimizer,
    patience: int,
    num_epochs: int,
    verbose: int
):
    best_loss = np.inf
    best_model = None
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        val_loss = _evaluate_model(model, val_loader, criterion)
        if verbose:
            print(f"Epoch {epoch}, val loss: {val_loss}, best loss: {best_loss}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch}")

            break

    if best_model:
        model.load_state_dict(best_model)

    return model

class _NNModel:
    """
    Simple wrapper for fitting a lightgbm model. See
    the lightgbm score function documentation for more details.
    """

    def __init__(
        self,
        learning_rate: float,
        num_layers: int,
        hidden_size: int,
        batch_size: int,
        seed: int,
        verbose: int,
        patience: int,
        num_epochs: int,
    ):
        self._model = None
        self._learning_rate = learning_rate
        self._num_layers = num_layers
        self._hidden_size = hidden_size
        self._batch_size = batch_size
        self._seed = seed
        self._verbose = verbose
        self._patience = patience
        self._num_epochs = num_epochs

    def fit(
        self,
        X: Float[np.ndarray, "batch_train x_dim"],
        y: Float[np.ndarray, "batch_train y_dim"],
        X_val: Float[np.ndarray, "batch_val x_dim"],
        y_val: Float[np.ndarray, "batch_val y_dim"]
    ):
        X = t.tensor(X).float()
        y = t.tensor(y).float()

        X_val = t.tensor(X_val).float()
        y_val = t.tensor(y_val).float()

        x_dim = X.shape[1]
        y_dim = y.shape[1]

        model = _MLPModule(self._num_layers, self._hidden_size, x_dim, y_dim)
        criterion = nn.MSELoss()

        train_loader = DataLoader(list(zip(X, y)), batch_size=self._batch_size, shuffle=True)
        val_loader = DataLoader(list(zip(X_val, y_val)), batch_size=self._batch_size, shuffle=False)

        optimizer = t.optim.Adam(model.parameters(), lr=self._learning_rate)

        model = _train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            patience=self._patience,
            num_epochs=self._num_epochs,
            verbose=self._verbose
        )
        model.eval()

    def predict(self, X: Float[np.ndarray, "batch x_dim"]) -> Float[np.ndarray, "batch y_dim"]:
        self._model.eval()
        with t.no_grad():
            return self._model(t.tensor(X)).numpy()

def _make_training_data(
    X: Float[np.ndarray, "batch x_dim"],
    y: Float[np.ndarray, "batch y_dim"],
    sde: DiffusionSDE,
    n_repeats: int,
    eval_percent: Optional[float],
    seed: Optional[int] = None,
):
    """
    Creates the training data for the score model. This functions assumes that
    1.  Score is parametrized as score(y, x, t) = GBT(y, x, t) / std(t)
    2.  The loss that we want to use is
        || std(t) * score(y_perturbed, x, t) - (mean(y, t) - y_perturbed)/std(t) ||^2
        Which corresponds to the standard denoising objective with weights std(t)**2
        This ends up meaning that we optimize
        || GBT(y_perturbed, x, t) - (-z)||^2
        where z is the noise added to y_perturbed.

    Returns:
    - predictors_train: X_train=[y_perturbed_train, x_train, t_train] for lgbm
    - predictors_val: X_val=[y_perturbed_val, x_val, t_val] for lgbm
    - predicted_train: y_train=[-z_train] for lgbm
    - predicted_val: y_val=[-z_val] for lgbm
    """
    EPS = 1e-5  # smallest step we can sample from
    T = sde.T
    if seed is not None:
        np.random.seed(seed)

    X_train, X_test, y_train, y_test = X, None, y, None
    predictors_train, predictors_val = None, None
    predicted_train, predicted_val = None, None

    if eval_percent is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=eval_percent, random_state=seed
        )

    # TRAINING DATA
    X_train = np.tile(X, (n_repeats, 1))
    y_train = np.tile(y, (n_repeats, 1))
    t_train = np.random.uniform(0, 1, size=(y_train.shape[0], 1)) * (T - EPS) + EPS
    z_train = np.random.normal(size=y_train.shape)

    train_mean, train_std = sde.get_mean_std_pt_given_y0(y_train, t_train)
    perturbed_y_train = train_mean + train_std * z_train

    predictors_train = np.concatenate([perturbed_y_train, X_train, t_train], axis=1)
    predicted_train = -1.0 * z_train

    # VALIDATION DATA
    if eval_percent is not None:
        t_val = np.random.uniform(0, 1, size=(y_test.shape[0], 1)) * (T - EPS) + EPS
        z_val = np.random.normal(size=(y_test.shape[0], y_test.shape[1]))

        val_mean, val_std = sde.get_mean_std_pt_given_y0(y_test, t_val)
        perturbed_y_val = val_mean + val_std * z_val
        predictors_val = np.concatenate(
            [perturbed_y_val, X_test, t_val.reshape(-1, 1)], axis=1
        )
        predicted_val = -1.0 * z_val

    return predictors_train, predictors_val, predicted_train, predicted_val


###################################################
# Main models
###################################################

# lightgbm score
class LightGBMScore(Score):
    def __init__(
        self,
        sde: DiffusionSDE,
        n_repeats: Optional[int] = 1,
        n_estimators: Optional[int] = 100,
        eval_percent: Optional[float] = None,
        early_stopping_rounds: Optional[int] = None,
        num_leaves: Optional[int] = 31,
        max_depth: Optional[int] = -1,
        learning_rate: Optional[float] = 0.1,
        max_bin: Optional[int] = 255,
        subsample_for_bin: Optional[int] = 200000,
        min_child_samples: Optional[int] = 20,
        subsample: Optional[float] = 1.0,
        subsample_freq: Optional[int] = 0,
        categorical_features: Optional[list[int]] = None,
        verbose: Optional[int] = 0,
        seed: Optional[int] = None,
        n_jobs: Optional[int] = -1,
    ) -> None:
        """
        Args:
        This model doesn't do any model checking or validation. It's assumed that
        that the main user is the `Treeffuser` class and that the user has already
        checked that the inputs are valid.

            Diffusion model args
            -------------------------------
            sde (SDE): A member from the SDE class specifying the sde that is implied
                by the score model.
            n_repeats (int): How many times to repeat the training dataset. i.e how
                many noisy versions of a point to generate for training.

            LightGBM args
            -------------------------------
            eval_percent (float): Percentage of the training data to use for validation.
                If `None`, no validation set is used.
            early_stopping_rounds (int): If `None`, no early stopping is performed. Otherwise,
                the model will stop training if no improvement is observed in the validation
                set for `early_stopping_rounds` consecutive iterations.
            n_estimators (int): Number of boosting iterations.
            num_leaves (int): Maximum tree leaves for base learners.
            max_depth (int): Maximum tree depth for base learners, <=0 means no limit.
            learning_rate (float): Boosting learning rate.
            max_bin (int): Max number of bins that feature values will be bucketed in. This
                is used for lightgbm's histogram binning algorithm.
            subsample_for_bin (int): Number of samples for constructing bins (can ignore).
            min_child_samples (int): Minimum number of data needed in a child (leaf). If
                less than this number, will not create the child.
            subsample (float): Subsample ratio of the training instance.
            subsample_freq (int): Frequence of subsample, <=0 means no enable.
                How often to subsample the training data.
            seed (int): Random seed.
            early_stopping_rounds (int): If `None`, no early stopping is performed. Otherwise,
                the model will stop training if no improvement is observed in the validation
            n_jobs (int): Number of parallel threads. If set to -1, the number is set to the
                number of available cores.
        """
        if early_stopping_rounds is not None:
            eval_percent = eval_percent if eval_percent is not None else 0.1

        # Diffusion model args
        self._sde = sde
        self._n_repeats = n_repeats
        self._eval_percent = eval_percent

        # LightGBM args
        self._lgbm_args = {
            "early_stopping_rounds": early_stopping_rounds,
            "n_estimators": n_estimators,
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "max_bin": max_bin,
            "subsample_for_bin": subsample_for_bin,
            "min_child_samples": min_child_samples,
            "subsample": subsample,
            "subsample_freq": subsample_freq,
            "categorical_features": categorical_features,
            "seed": seed,
            "verbose": verbose,
            "n_jobs": n_jobs,
        }

        # Other stuff part of internal state
        self.models = None  # Convention inputs are (y, x, t)
        self.is_fitted = False

    def score(
        self,
        y: Float[np.ndarray, "batch y_dim"],
        X: Float[np.ndarray, "batch x_dim"],
        t: Int[np.ndarray, "batch 1"],
    ) -> Float[np.ndarray, "batch y_dim"]:
        scores = []
        predictors = np.concatenate([y, X, t], axis=1)
        _, std = self._sde.get_mean_std_pt_given_y0(y, t)
        for i in range(y.shape[-1]):
            # The score is parametrized: score(y, x, t) = GBT(y, x, t) / std(t)
            score_p = self.models[i].predict(predictors, num_threads=self._lgbm_args["n_jobs"])
            score = score_p / std[:, i]
            scores.append(score)
        return np.array(scores).T

    def fit(
        self,
        X: Float[np.ndarray, "batch x_dim"],
        y: Float[np.ndarray, "batch y_dim"],
    ):
        """
        Fit the score model to the data.

        Args:
            X: input data
            y: target data
            n_repeats: How many times to repeat the training dataset.
            likelihood_reweighting: Whether to reweight the likelihoods.
            likelihood_weighting: If `True`, weight the mixture of score
                matching losses according to https://arxiv.org/abs/2101.09258;
                otherwise use the weighting recommended in song's SDEs paper.
        """
        y_dim = y.shape[1]

        lgb_X_train, lgb_X_val, lgb_y_train, lgb_y_val = _make_training_data(
            X=X,
            y=y,
            sde=self._sde,
            n_repeats=self._n_repeats,
            eval_percent=self._eval_percent,
            seed=self._lgbm_args["seed"],
        )

        models = []
        for i in range(y_dim):
            lgb_y_val_i = lgb_y_val[:, i] if lgb_y_val is not None else None
            score_model_i = _fit_one_lgbm_model(
                X=lgb_X_train,
                y=lgb_y_train[:, i],
                X_val=lgb_X_val,
                y_val=lgb_y_val_i,
                **self._lgbm_args,
            )
            models.append(score_model_i)
        self.models = models
