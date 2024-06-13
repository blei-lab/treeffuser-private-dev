"""
This file should contain a general abstraction of the score models and
should function as a wrapper for different models we might want to use.
"""
import abc

import abc
from typing import List
from typing import Optional

import lightgbm as lgb
import numpy as np
from jaxtyping import Float
from jaxtyping import Int
from sklearn.model_selection import train_test_split

from treeffuser.sde import DiffusionSDE
from treeffuser._score_models._base import ScoreModel


import numpy as np
from jaxtyping import Float
from jaxtyping import Int


class ScoreModel(abc.ABC):
    @abc.abstractmethod
    def score(
        self,
        X: Float[np.ndarray, "batch x_dim"],
        y: Float[np.ndarray, "batch y_dim"],
        t: Int[np.ndarray, "batch"],
    ):

        pass

    @abc.abstractmethod
    def fit(
        self,
        X: Float[np.ndarray, "batch x_dim"],
        y: Float[np.ndarray, "batch y_dim"],
        sde: DiffusionSDE,
        cat_idx: Optional[List[int]] = None,
    ):
        pass
