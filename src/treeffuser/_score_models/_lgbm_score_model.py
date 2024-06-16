"""
Contains different score models to be used to approximate the score of a given SDE.
"""

from typing import List
from typing import Optional

import lightgbm as lgb
import numpy as np
from jaxtyping import Float
from jaxtyping import Int

from treeffuser._score_models._base import ScoreModel
from treeffuser._score_models._utils import make_training_data
from treeffuser.sde import DiffusionSDE

###################################################
# Helper functions
###################################################


def _fit_one_lgbm_model(
    X: Float[np.ndarray, "batch x_dim"],
    y: Float[np.ndarray, "batch y_dim"],
    X_val: Float[np.ndarray, "batch x_dim"],
    y_val: Float[np.ndarray, "batch y_dim"],
    seed: int,
    verbose: int,
    cat_idx: Optional[List[int]] = None,
    n_jobs: int = -1,
    early_stopping_rounds: Optional[int] = None,
    **lgbm_args,
) -> lgb.LGBMRegressor:
    """
    Simple wrapper for fitting a lightgbm model. See
    the lightgbm score function documentation for more details.
    """
    callbacks = None
    if early_stopping_rounds is not None:
        callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=verbose > 0)]

    model = lgb.LGBMRegressor(
        random_state=seed,
        verbose=verbose,
        n_jobs=n_jobs,
        linear_tree=False,
        **lgbm_args,
    )
    eval_set = None if X_val is None else (X_val, y_val)
    if cat_idx is None:
        cat_idx = "auto"
    model.fit(
        X=X,
        y=y,
        eval_set=eval_set,
        callbacks=callbacks,
        categorical_feature=cat_idx,
    )
    return model


###################################################
# Main models
###################################################


class LightGBMScoreModel(ScoreModel):
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
    **lgbm_args
        Additional arguments to pass to the LightGBM model. See the LightGBM documentation for more
        information. E.g. `early_stopping_rounds`, `n_estimators`, `learning_rate`, `max_depth`,

    Attributes
    ----------
    n_estimators_true : List[int]
        The true number of trees in each model (in case early stopping is used).
    """

    def __init__(
        self,
        n_repeats: Optional[int] = 10,
        eval_percent: float = 0.1,
        n_jobs: Optional[int] = -1,
        seed: Optional[int] = None,
        **lgbm_args,
    ) -> None:
        self.n_repeats = n_repeats
        self.eval_percent = eval_percent
        self.n_jobs = n_jobs
        self.seed = seed

        self._lgbm_args = lgbm_args
        self.sde = None
        self.models = None  # Convention inputs are (y, x, t)
        self.n_estimators_true = None

    def score(
        self,
        y: Float[np.ndarray, "batch y_dim"],
        X: Float[np.ndarray, "batch x_dim"],
        t: Int[np.ndarray, "batch 1"],
    ) -> Float[np.ndarray, "batch y_dim"]:
        if self.sde is None:
            raise ValueError("The model has not been fitted yet.")

        scores = []
        predictors = np.concatenate([y, X, t], axis=1)
        _, std = self.sde.get_mean_std_pt_given_y0(y, t)
        for i in range(y.shape[-1]):
            # The score is parametrized: score(y, x, t) = GBT(y, x, t) / std(t)
            score_p = self.models[i].predict(predictors, num_threads=self.n_jobs)
            score = score_p / std[:, i]
            scores.append(score)
        return np.array(scores).T

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
        y_dim = y.shape[1]
        self.sde = sde

        lgb_X_train, lgb_X_val, lgb_y_train, lgb_y_val, cat_idx = make_training_data(
            X=X,
            y=y,
            sde=self.sde,
            n_repeats=self.n_repeats,
            eval_percent=self.eval_percent,
            cat_idx=cat_idx,
            seed=self.seed,
        )

        models = []
        for i in range(y_dim):
            lgb_y_val_i = lgb_y_val[:, i] if lgb_y_val is not None else None
            score_model_i = _fit_one_lgbm_model(
                X=lgb_X_train,
                y=lgb_y_train[:, i],
                X_val=lgb_X_val,
                y_val=lgb_y_val_i,
                cat_idx=cat_idx,
                seed=self.seed,
                n_jobs=self.n_jobs,
                **self._lgbm_args,
            )
            models.append(score_model_i)
        self.models = models

        # collect the true number of trees learned by each model
        self.n_estimators_true = [model.n_estimators_ for model in self.models]
