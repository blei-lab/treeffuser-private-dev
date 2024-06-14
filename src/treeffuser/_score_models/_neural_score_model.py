"""
This file should contain a general abstraction of the score models and
should function as a wrapper for different models we might want to use.
"""

from typing import Callable
from typing import List
from typing import Optional

import numpy as np
import torch as t
import torch.nn as nn
from jaxtyping import Float
from jaxtyping import Int
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from treeffuser._score_models._base import ScoreModel
from treeffuser._score_models._utils import make_training_data
from treeffuser.scaler import ScalerMixedTypes
from treeffuser.sde import DiffusionSDE

###################################################
# Helper functions and classes
###################################################


class _MLPModule(nn.Module):
    def __init__(self, n_layers: int, hidden_size: int, input_size: int, output_size: int):
        """
        Simple MLP model with ReLU activation functions.
        """
        super().__init__()
        layers = []

        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())

        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_size, output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x: Float[t.Tensor, "batch x_dim"]) -> Float[t.Tensor, "batch y_dim"]:
        return self.model(x)


def _evaluate_model(model: nn.Module, data_loader: DataLoader, criterion: Callable) -> float:
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
    early_stopping_rounds: int,
    n_epochs: int,
    verbose: int,
):
    best_loss = np.inf
    best_model = None
    best_iter = 0
    patience_counter = 0

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()

        train_loss = epoch_loss / len(train_loader)
        val_loss = _evaluate_model(model, val_loader, criterion)

        if verbose:
            print(
                f"Epoch {epoch}, train_loss {train_loss}, val loss: {val_loss}, best loss: {best_loss}"
            )

        if val_loss < best_loss:
            best_loss = val_loss
            best_iter = epoch
            best_model = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_rounds:
            if verbose:
                msg = "Early stopping at epoch {}, best loss: {}, best iter: {}"
                print(msg.format(epoch, best_loss, best_iter))
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
        n_layers: int,
        hidden_size: int,
        batch_size: int,
        seed: int,
        verbose: int,
        early_stopping_rounds: int,
        n_epochs: int,
    ):
        self._model = None
        self._learning_rate = learning_rate
        self._n_layers = n_layers
        self._hidden_size = hidden_size
        self._batch_size = batch_size
        self._seed = seed
        self._verbose = verbose
        self._early_stopping_rounds = early_stopping_rounds
        self._n_epochs = n_epochs

        self.x_scaler = None
        self.y_scaler = None

    def fit(
        self,
        X: Float[np.ndarray, "batch_train x_dim"],
        y: Float[np.ndarray, "batch_train y_dim"],
        X_val: Float[np.ndarray, "batch_val x_dim"],
        y_val: Float[np.ndarray, "batch_val y_dim"],
    ):
        self.x_scaler = ScalerMixedTypes(scaler=MinMaxScaler())
        self.y_scaler = ScalerMixedTypes(scaler=MinMaxScaler())

        X = self.x_scaler.fit_transform(X)
        y = self.y_scaler.fit_transform(y)
        X_val = self.x_scaler.transform(X_val)
        y_val = self.y_scaler.transform(y_val)

        if self._seed is not None:
            t.manual_seed(self._seed)

        X = t.tensor(X).float()
        y = t.tensor(y).float()

        X_val = t.tensor(X_val).float()
        y_val = t.tensor(y_val).float()

        x_dim = X.shape[1]
        y_dim = y.shape[1]

        model = _MLPModule(self._n_layers, self._hidden_size, x_dim, y_dim)
        criterion = nn.MSELoss(reduction="mean")

        train_loader = DataLoader(list(zip(X, y)), batch_size=self._batch_size, shuffle=True)
        val_loader = DataLoader(
            list(zip(X_val, y_val)), batch_size=self._batch_size, shuffle=False
        )

        optimizer = t.optim.Adam(model.parameters(), lr=self._learning_rate)

        model = _train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            early_stopping_rounds=self._early_stopping_rounds,
            n_epochs=self._n_epochs,
            verbose=self._verbose,
        )
        model.eval()
        self._model = model

    def predict(self, X: Float[np.ndarray, "batch x_dim"]) -> Float[np.ndarray, "batch y_dim"]:
        X = self.x_scaler.transform(X)
        self._model.eval()
        with t.no_grad():
            X = t.tensor(X).float()
            y = self._model(X).detach().numpy()
            y = self.y_scaler.inverse_transform(y)

            return y


###################################################
# Main models
###################################################


class NeuralScoreModel(ScoreModel):
    """
    A score model that uses a LightGBM model (trees) to approximate the score of a given SDE.

    Parameters
    ----------
    n_repeats : int
        How many times to repeat the training dataset when fitting the score. That is, how many
        noisy versions of a point to generate for training.
    eval_percent : float
        Percentage of the training data to use for validation for optional early stopping. It is
        ignored if `early_stopping_rounds` is not set in the `lgbm_args`.
    n_jobs : int
        LightGBM: Number of parallel threads. If set to -1, the number is set to the number of available cores.
    seed : int
        Random seed for generating the training data and fitting the model.
    verbose : int
        Verbosity of the score model.
    use_separate_models: bool
        Whether to train a separate model for each output dimension or a single model for all dimensions.
    **nn_args:
        Additional arguments to pass to the NN model.
        See the _NNModel class for more information.
    """

    def __init__(
        self,
        n_repeats: Optional[int],
        eval_percent: float,
        n_jobs: Optional[int],
        use_separate_models: bool,
        seed: Optional[int] = 0,
        **nn_args,
    ) -> None:
        self.n_repeats = n_repeats
        self.eval_percent = eval_percent
        self.n_jobs = n_jobs
        self.seed = seed
        self.use_separate_models = use_separate_models

        self._nn_args = nn_args
        self.sde = None
        self.models = None  # Convention inputs are (x, y, t)

    def score(
        self,
        y: Float[np.ndarray, "batch y_dim"],
        X: Float[np.ndarray, "batch x_dim"],
        t: Int[np.ndarray, "batch 1"],
    ) -> Float[np.ndarray, "batch y_dim"]:
        if self.sde is None:
            raise ValueError("The model has not been fitted yet.")

        scores_p = []

        predictors = np.concatenate([y, X, t], axis=1)
        _, std = self.sde.get_mean_std_pt_given_y0(y, t)

        for i in range(len(self.models)):
            model = self.models[i]
            score_p = model.predict(predictors)
            scores_p.append(score_p)

        scores_p = np.array(scores_p)

        # This handles the separate models case
        if scores_p.ndim == 2:
            scores_p = scores_p.T
        elif scores_p.ndim == 3:  # remove last dimension
            scores_p = scores_p.squeeze(-1).T

        scores = scores_p / std
        return scores

    def fit(
        self,
        X: Float[np.ndarray, "batch x_dim"],
        y: Float[np.ndarray, "batch y_dim"],
        sde: DiffusionSDE,
        cat_idx: Optional[List[int]] = None,
    ):
        """
        Fit the score model to the data and the given SDE.

        Parameters
        ----------
        X : Float[np.ndarray, "batch x_dim"]
            The input data.
        y : Float[np.ndarray, "batch y_dim"]
            The true output values.
        sde : DiffusionSDE
            The SDE that the model is supposed to approximate the score of.
        cat_idx : Optional[List[int]]
            List of indices of categorical features in the input data. If `None`, all features are
            assumed to be continuous.
        """
        if len(y.shape) != 2:
            raise ValueError("y should have shape (batch_train, y_dim)")

        self.sde = sde
        nn_X_train, nn_X_val, nn_y_train, nn_y_val, cat_idx = make_training_data(
            X=X,
            y=y,
            sde=self.sde,
            n_repeats=self.n_repeats,
            eval_percent=self.eval_percent,
            cat_idx=cat_idx,
            seed=self.seed,
        )
        if self.use_separate_models:
            self._fit_separate(nn_X_train, nn_y_train, nn_X_val, nn_y_val, cat_idx)
        else:
            self._fit_single(nn_X_train, nn_y_train, nn_X_val, nn_y_val, cat_idx)

    def _fit_separate(
        self,
        nn_X_train: Float[np.ndarray, "batch_train x_dim"],
        nn_y_train: Float[np.ndarray, "batch_train y_dim"],
        nn_X_val: Float[np.ndarray, "batch_val x_dim"],
        nn_y_val: Float[np.ndarray, "batch_val y_dim"],
        cat_idx: Optional[List[int]] = None,
    ):
        """
        Fit the score model to the data and the given SDE.
        All output dimensions share the same model.

        Parameters
        ----------
        nn_X_train : Float[np.ndarray, "batch_train x_dim"]
            The input data.
        nn_y_train : Float[np.ndarray, "batch_train y_dim"]
            The true output values.
        nn_X_val : Float[np.ndarray, "batch_val x_dim"]
            The input data.
        nn_y_val : Float[np.ndarray, "batch_val y_dim"]
            The true output values.
        cat_idx : Optional[List[int]]
            List of indices of categorical features in the input data. If `None`, all features are
            assumed to be continuous.
        """
        # check y shape (batch_train, y_dim)

        models = []
        y_dim = nn_y_train.shape[1]

        for i in range(y_dim):
            model = _NNModel(**self._nn_args, seed=self.seed)
            model.fit(
                nn_X_train,
                nn_y_train[:, i].reshape(-1, 1),  # (batch_train, 1)
                nn_X_val,
                nn_y_val[:, i].reshape(-1, 1),  # (batch_val, 1)
            )
            models.append(model)
        self.models = models

    def _fit_single(
        self,
        nn_X_train: Float[np.ndarray, "batch_train x_dim"],
        nn_y_train: Float[np.ndarray, "batch_train y_dim"],
        nn_X_val: Float[np.ndarray, "batch_val x_dim"],
        nn_y_val: Float[np.ndarray, "batch_val y_dim"],
        cat_idx: Optional[List[int]] = None,
    ):
        """
        Fit the score model to the data and the given SDE.
        All output dimensions share the same model.

        Parameters
        ----------
        nn_X_train : Float[np.ndarray, "batch_train x_dim"]
            The input data.
        nn_y_train : Float[np.ndarray, "batch_train y_dim"]
            The true output values.
        nn_X_val : Float[np.ndarray, "batch_val x_dim"]
            The input data.
        nn_y_val : Float[np.ndarray, "batch_val y_dim"]
            The true output values.
        cat_idx : Optional[List[int]]
            List of indices of categorical features in the input data. If `None`, all features are
            assumed to be continuous.
        """
        model = _NNModel(**self._nn_args, seed=self.seed)
        model.fit(nn_X_train, nn_y_train, nn_X_val, nn_y_val)
        self.models = [model]
