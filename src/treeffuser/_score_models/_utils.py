from jaxtyping import Float
from jaxtyping import Int
import numpy as np
from typing import List
from typing import Optional

from sklearn.model_selection import train_test_split
from treeffuser.sde import DiffusionSDE



def make_training_data(
    X: Float[np.ndarray, "batch x_dim"],
    y: Float[np.ndarray, "batch y_dim"],
    sde: DiffusionSDE,
    n_repeats: int,
    eval_percent: Optional[float],
    cat_idx: Optional[List[int]] = None,
    seed: Optional[int] = None,
):
    """
    Creates the training data for the score model. This functions assumes that
    1.  Score is parametrized as score(x, y, t) = GBT(x, y, t) / std(t)
    2.  The loss that we want to use is
        || std(t) * score(y_perturbed, x, t) - (mean(y, t) - y_perturbed)/std(t) ||^2
        Which corresponds to the standard denoising objective with weights std(t)**2
        This ends up meaning that we optimize
        || GBT(y_perturbed, x, t) - (-z)||^2
        where z is the noise added to y_perturbed.

    Returns:
    - predictors_train: X_train=[x_train, y_perturbed_train, t_train] for lgbm
    - predictors_val: X_val=[x_val, y_perturbed_val, t_val] for lgbm
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

    predictors_train = np.concatenate([X_train, perturbed_y_train, t_train], axis=1)
    predicted_train = -1.0 * z_train

    # VALIDATION DATA
    if eval_percent is not None:
        t_val = np.random.uniform(0, 1, size=(y_test.shape[0], 1)) * (T - EPS) + EPS
        z_val = np.random.normal(size=(y_test.shape[0], y_test.shape[1]))

        val_mean, val_std = sde.get_mean_std_pt_given_y0(y_test, t_val)
        perturbed_y_val = val_mean + val_std * z_val
        predictors_val = np.concatenate(
            [X_test, perturbed_y_val, t_val.reshape(-1, 1)], axis=1
        )
        predicted_val = -1.0 * z_val

    # cat_idx is not changed
    return predictors_train, predictors_val, predicted_train, predicted_val, cat_idx


