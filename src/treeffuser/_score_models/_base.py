"""
This file should contain a general abstraction of the score models and
should function as a wrapper for different models we might want to use.
"""
import abc


import numpy as np
from jaxtyping import Float
from jaxtyping import Int

class Score(abc.ABC):
    @abc.abstractmethod
    def score(
        self,
        X: Float[np.ndarray, "batch x_dim"],
        y: Float[np.ndarray, "batch y_dim"],
        t: Int[np.ndarray, "batch"],
    ):

        pass

    @abc.abstractmethod
    def fit(self, X: Float[np.ndarray, "batch x_dim"], y: Float[np.ndarray, "batch y_dim"]):
        pass
