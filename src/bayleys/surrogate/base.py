from abc import abstractmethod, ABC
from dataclasses import dataclass
import numpy as np
import torch
from torch.optim import Adam, AdamW, LBFGS

from ..utils import AbstractConfig

from ..encoders.base import MoleculeEncoder


OPTIMIZER_FACTORY = {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW, "LBFGS": torch.optim.LBFGS}


@dataclass
class ModelArchitecture(AbstractConfig, ABC):
    """
    Abstract base class for model architectures. Must contain all hyperparameters that are related to the surrogate
    model architecture (e.g. number of layers, hidden dimensions, likelihood type, etc.).
    """
    pass


@dataclass
class TrainingConfig(AbstractConfig):
    """
    Dataclass that holds training configuration parameters for surrogate models.
    """

    optimizer: str = "Adam"

    batch_size: int = 128
    num_epochs: int = 1500
    learning_rate: float = 1E-4
    weight_decay: float = 1E-5
    early_stopping: bool = True
    early_stopping_patience: int = 50
    n_jobs: int = -2

    def __post_init__(self):
        """
        Validates the training configuration after initialization by performing the following checks:
          - Ensures that the specified optimizer exists in the OPTIMIZER_FACTORY.
        """
        if self.optimizer not in OPTIMIZER_FACTORY:
            raise ValueError(f"Optimizer '{self.optimizer}' is not recognized!")

    def get_optimizer(self, model_parameters):
        """
        Instantiates and returns the optimizer based on the configuration.
        """
        if self.optimizer == "Adam":
            return Adam(model_parameters, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer == "AdamW":
            return AdamW(model_parameters, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer == "LBFGS":
            return LBFGS(model_parameters, line_search_fn="strong_wolfe")
        else:
            raise ValueError(f"Optimizer '{self.optimizer}' is not recognized!")


class Model(ABC):
    """
    Abstract base class for surrogate models.
    """
    name: str

    default_architecture: ModelArchitecture
    default_training_config: TrainingConfig

    def __init__(
            self,
            architecture: ModelArchitecture,
            training_config: TrainingConfig,
            encoder: MoleculeEncoder,
            static_embeddings: bool = True,
            random_seed: int = 42,
            **kwargs
    ):

        """
        Abstract base class for surrogate models.

        Args:
            architecture (ModelArchitecture): Architecture configuration for the model. The respective subclass must be
            ️                                 defined for each model type.
            training_config (TrainingConfig): Training configuration for the model.
            encoder (MoleculeEncoder): Molecule encoder to be used by the model.
            random_seed (int): Random seed for reproducibility.
            static_embeddings (bool): Whether to use static precomputed embeddings or adapt them on-the-fly.
        """
        self.architecture = architecture
        self.training_config = training_config
        self.encoder = encoder
        self.static_embeddings = static_embeddings
        self.random_seed = random_seed

    @abstractmethod
    def train_model(
            self,
            train_smiles: np.ndarray,
            train_labels: torch.Tensor,
            train_embeddings: torch.Tensor = None
    ):
        """
        Trains the model on the provided data.

        Args:
            train_smiles (np.ndarray): Array of SMILES strings for training data.
            train_labels (torch.Tensor): Tensor of target labels for training data.
            train_embeddings (torch.Tensor, optional): Tensor of precomputed embeddings for training data. Only used if
                                                       static_embeddings is True.
        """
        raise NotImplementedError

    @abstractmethod
    def retrain_model(
            self,
            train_smiles: np.ndarray,
            train_labels: torch.Tensor,
            train_embeddings: torch.Tensor = None
    ):
        """
        Retrains the model on the provided data.

        Args:
            train_smiles (np.ndarray): Array of SMILES strings for training data.
            train_labels (torch.Tensor): Tensor of target labels for training data.
            train_embeddings (torch.Tensor, optional): Tensor of precomputed embeddings for training data. Only used if
                                                       static_embeddings is True.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(
            self,
            embeddings: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Predicts the target labels for the provided SMILES strings.

        Args:
            embeddings (torch.Tensor, optional): Tensor of precomputed embeddings for prediction data. Only used if
                                                 static_embeddings is True.

        Returns:
            torch.Tensor: Predicted target labels.
        """
        raise NotImplementedError
