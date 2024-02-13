"""
This file should contain a general abstraction of the score models and
should function as a wrapper for different models we might want to use.

The idea is to "hide" the particular tree we want to use so that
we can easily switch between different models without having to change
the rest of the code.
"""

import abc
from typing import Float

import numpy as np

from treeffuser.sde import SDE


class Score(abc.ABC):
    @abc.abstractmethod
    def score(self, X: Float[np.ndarray, "batch x_dim"], y: Float[np.ndarray, "batch y_dim"]):
        pass

    @abc.abstractmethod
    def fit(self, X: Float[np.ndarray, "batch x_dim"], y: Float[np.ndarray, "batch y_dim"]):
        pass


# lightgbm score
class LightGBMScore(Score):
    def __init__(self, sde: SDE):
        self.sde = sde

    def score(self, X: Float[np.ndarray, "batch x_dim"], y: Float[np.ndarray, "batch y_dim"]):
        pass

    def fit(self, X: Float[np.ndarray, "batch x_dim"], y: Float[np.ndarray, "batch y_dim"]):
        pass
