from typing import Literal
from typing import Optional

from treeffuser._base_tabular_diffusion import BaseTabularDiffusion
from treeffuser._score_models import NeuralScoreModel
from treeffuser._score_models import ScoreModel
from treeffuser.sde import DiffusionSDE
from treeffuser.sde import get_diffusion_sde


class NNffuser(BaseTabularDiffusion):
    """
    A standard diffusion model with a neural network score model.
    """

    def __init__(
        self,
        n_repeats: int = 10,
        n_epochs: int = 100,
        early_stopping_rounds: int = 10,
        eval_percent: float = 0.1,
        use_separate_models: bool = True,
        learning_rate: float = 0.01,
        n_layers: int = 2,
        hidden_size: int = 10,
        batch_size: int = 32,
        n_jobs: int = -1,
        sde_name: str = "vesde",
        sde_initialize_from_data: bool = False,
        sde_hyperparam_min: Optional[float | Literal["default"]] = None,
        sde_hyperparam_max: Optional[float | Literal["default"]] = None,
        seed: Optional[int] = None,
        verbose: int = 0,
    ):
        """
        n_repeats : int
            How many times to repeat the training dataset when fitting the score. That is, how many
            noisy versions of a point to generate for training.
        n_epochs : int
            NN: Maximum number of epochs to train the neural network for. Due to early stopping,
            the actual number of epochs may be less.
        early_stopping_rounds : int
            NN: If `None`, no early stopping is performed. Otherwise, the model will stop training
            if no improvement is observed in the validation set for `early_stopping_rounds` consecutive
            epochs.
        eval_percent : float
            NN: Percentage of the training data to use for validation if `early_stopping_rounds`
            is not `None`.
        use_separate_models : bool
            NN: Whether to use separate models for each dimension of the score. If `True`, a separate
            model is trained for each dimension of the score. If `False`, a single model is trained
            for all dimensions of the score.
        learning_rate : float
            NN: Learning rate for the neural network.
        n_layers : int
            NN: Number of hidden layers in the neural network.
        hidden_size : int
            NN: Number of hidden units in each hidden layer of the neural network.
        batch_size : int
            NN: Batch size for training the neural network.
        n_jobs : int
            NN: Number of parallel threads. If set to -1, the number is set to the number of available cores.
        sde_name : str
            SDE: Name of the SDE to use. See `treeffuser.sde.get_diffusion_sde` for available SDEs.
        sde_initialize_from_data : bool
            SDE: Whether to initialize the SDE from the data. If `True`, the SDE hyperparameters are
            initialized with a heuristic based on the data (see `treeffuser.sde.initialize.py`).
            Otherwise, sde_hyperparam_min and sde_hyperparam_max are used. (default: False)
        sde_hyperparam_min : float or "default"
            SDE: The scale of the SDE at t=0 (see `VESDE`, `VPSDE`, `SubVPSDE`).
        sde_hyperparam_max : float or "default"
            SDE: The scale of the SDE at t=T (see `VESDE`, `VPSDE`, `SubVPSDE`).
        seed : int
            Random seed for generating the training data and fitting the model.
        verbose : int
            Verbosity of the score model.
        """
        super().__init__(
            sde_initialize_from_data=sde_initialize_from_data,
        )
        self.sde_name = sde_name
        self.n_repeats = n_repeats
        self.n_epochs = n_epochs
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_percent = eval_percent
        self.use_separate_models = use_separate_models
        self.learning_rate = learning_rate
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_jobs = n_jobs
        self.sde_hyperparam_min = sde_hyperparam_min
        self.sde_hyperparam_max = sde_hyperparam_max
        self.seed = seed
        self.verbose = verbose

    def get_new_sde(self) -> DiffusionSDE:
        sde_cls = get_diffusion_sde(self.sde_name)
        sde_kwargs = {}
        if self.sde_hyperparam_min is not None:
            sde_kwargs["hyperparam_min"] = self.sde_hyperparam_min
        if self.sde_hyperparam_max is not None:
            sde_kwargs["hyperparam_max"] = self.sde_hyperparam_max
        sde = sde_cls(**sde_kwargs)
        return sde

    def get_new_score_model(self) -> ScoreModel:
        score_model = NeuralScoreModel(
            n_repeats=self.n_repeats,
            eval_percent=self.eval_percent,
            early_stopping_rounds=self.early_stopping_rounds,
            n_epochs=self.n_epochs,
            learning_rate=self.learning_rate,
            n_layers=self.n_layers,
            hidden_size=self.hidden_size,
            batch_size=self.batch_size,
            use_separate_models=self.use_separate_models,
            verbose=self.verbose,
            seed=self.seed,
            n_jobs=self.n_jobs,
        )
        return score_model
