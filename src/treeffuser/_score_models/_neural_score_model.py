"""
This file should contain a general abstraction of the score models and
should function as a wrapper for different models we might want to use.
"""

from typing import Callable
from typing import List
from typing import Optional

import numpy as np

###################################################
# Helper functions and classes
###################################################
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


class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Initialize shadow parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Update shadow parameters
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        # Save the current parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                # Replace model parameters with shadow parameters
                param.data = self.shadow[name]

    def restore(self):
        # Restore the original parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


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


class _CardLikeMLPModule(nn.Module):
    def __init__(self, n_layers: int, hidden_size: int, input_size: int, output_size: int):
        """
        Simple MLP model with ReLU activation functions. Using Card like architecture.
        In particular, this assumes that the input is of the form [y, x, t]
        and an embedding of t is added after each layer.
        """
        super().__init__()
        layers = []
        t_embeddings = []

        layers.append(nn.Linear(input_size - 1, hidden_size))
        t_embeddings.append(nn.Linear(1, hidden_size))

        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            t_embeddings.append(nn.Linear(1, hidden_size))

        layers.append(nn.Linear(hidden_size, output_size))

        self.layers = nn.ModuleList(layers)
        self.t_embeddings = nn.ModuleList(t_embeddings)

    def forward(self, x: Float[t.Tensor, "batch x_dim"]) -> Float[t.Tensor, "batch y_dim"]:
        yx = x[:, :-1]
        t = x[:, -1].unsqueeze(-1)
        out = yx
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if i < len(self.layers) - 1:
                out += self.t_embeddings[i](t)
                out = nn.ReLU()(out)
        return out


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
    max_evals: int,
    decay: float,
    eval_freq: int,
    verbose: int,
):
    best_loss = np.inf
    best_model = None
    best_iter = 0
    patience_counter = 0

    n_iters = 0
    eval_round_train_loss = 0.0

    ema = EMA(model, decay=decay)

    while True:
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            eval_round_train_loss += loss.item()
            optimizer.step()
            ema.update()

            n_iters += 1

            if n_iters % eval_freq == 0:
                ema.apply_shadow()
                val_loss = _evaluate_model(model, val_loader, criterion)
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model = model.state_dict()
                    best_iter = n_iters
                    patience_counter = 0
                else:
                    patience_counter += 1

                if verbose:
                    msg = "Iter {}, val loss: {}, train loss: {}"
                    eval_round_train_loss /= eval_freq
                    print(msg.format(n_iters, val_loss, eval_round_train_loss))

                if patience_counter >= early_stopping_rounds or n_iters >= max_evals:
                    break

                eval_round_train_loss = 0.0
                ema.restore()

        if patience_counter >= early_stopping_rounds or n_iters >= max_evals:
            break

    ema.apply_shadow()
    if verbose:
        msg = "Best model found at iter {}, with val loss: {}"
        print(msg.format(best_iter, best_loss))
        if n_iters >= max_evals:
            print("Training stopped because max evals reached.")
        if patience_counter >= early_stopping_rounds:
            print("Training stopped because of early stopping.")

    if best_model:
        model.load_state_dict(best_model)

    return model


class _NNModel:
    """
    A simple neural network model for regression with the option to implement
    a card-like architecture specialized for diffusion models.

    For further details on the card-like architecture,
    see appendix of the paper:
        https://arxiv.org/pdf/2206.07275
    """

    def __init__(
        self,
        learning_rate: float,
        n_layers: int,
        hidden_size: int,
        batch_size: int,
        verbose: int,
        eval_freq: int,
        decay: float,
        early_stopping_rounds: int,
        max_evals: int,
        card_like: bool = False,
        weight_decay: float = 0.0,
        seed: int = 0,
    ):
        """
        Simple neural network model for regression. Specialized for diffusion models.

        learning_rate : float
            The learning rate for the optimizer.
        n_layers : int
            The number of hidden layers in the neural network.
        hidden_size : int
            The size of the hidden layers.
        batch_size : int
            The batch size for training.
        seed : int
            The random seed for the model.
        verbose : int
            Verbosity of the model.
        eval_freq : int
            After how many iterations to perform the evaluation of the
            validation set.
        decay : float
            The decay rate for the exponential moving average.
        early_stopping_rounds : int
            The number of evaluations without improvement before stopping.
        max_evals : int
            The maximum number of evaluations. This would be the equivalent
            to the maximum number of epochs in a normal training loop.
        card_like : bool
            Whether to use the card-like architecture. See `_CardLikeMLPModule`.
            for more information.
        weight_decay : float
        seed : int
        """

        self._model = None
        self._learning_rate = learning_rate
        self._n_layers = n_layers
        self._hidden_size = hidden_size
        self._batch_size = batch_size
        self._seed = seed
        self._verbose = verbose
        self._early_stopping_rounds = early_stopping_rounds
        self._eval_freq = eval_freq
        self._max_evals = max_evals
        self._weight_decay = weight_decay
        self._card_like = card_like
        self._decay = decay

        self.x_scaler = None
        self.y_scaler = None

    def fit(
        self,
        X: Float[np.ndarray, "batch_train x_dim"],
        y: Float[np.ndarray, "batch_train y_dim"],
        X_val: Float[np.ndarray, "batch_val x_dim"],
        y_val: Float[np.ndarray, "batch_val y_dim"],
    ):
        self.x_scaler = ScalerMixedTypes(scaler=MinMaxScaler((-1, 1)))
        self.y_scaler = ScalerMixedTypes(scaler=MinMaxScaler((-1, 1)))

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

        if self._card_like:
            model = _CardLikeMLPModule(self._n_layers, self._hidden_size, x_dim, y_dim)
        else:
            model = _MLPModule(self._n_layers, self._hidden_size, x_dim, y_dim)

        criterion = nn.MSELoss(reduction="mean")

        train_loader = DataLoader(list(zip(X, y)), batch_size=self._batch_size, shuffle=True)
        val_loader = DataLoader(
            list(zip(X_val, y_val)), batch_size=self._batch_size, shuffle=False
        )

        optimizer = t.optim.Adam(
            model.parameters(),
            lr=self._learning_rate,
            weight_decay=self._weight_decay,
            amsgrad=True,
        )

        model = _train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            early_stopping_rounds=self._early_stopping_rounds,
            max_evals=self._max_evals,
            eval_freq=self._eval_freq,
            verbose=self._verbose,
            decay=self._decay,
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
        self.models = None  # Convention inputs are (y, x, t)

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
            score_p = model.predict(predictors) / std[:, i]
            scores_p.append(score_p)

        scores_p = np.array(scores_p)

        # This handles the separate models case
        if scores_p.ndim == 2:
            scores_p = scores_p.T
        elif scores_p.ndim == 3:  # remove last dimension
            scores_p = scores_p[..., 0].T

        return scores_p

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
