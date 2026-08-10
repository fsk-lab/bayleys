from typing import Literal
from dataclasses import dataclass
import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor

from .base import ModelArchitecture, TrainingConfig, Model

from ..encoders.base import MoleculeEncoder


@dataclass
class RandomForestArchitecture(ModelArchitecture):
    """
    Dataclass that holds architecture parameters for Random Forest surrogate models.
    """
    n_estimators: int = 200
    max_depth: int = None
    min_samples_split: int = 2
    max_features: Literal["sqrt", "log2", None] = "sqrt"


class RandomForestModel(Model):
    """
    Random Forest surrogate model for molecular property prediction and uncertainty estimation (via ensemble variance).
    """
    name = "RF"

    default_architecture = RandomForestArchitecture()
    default_training_config = TrainingConfig()

    def __init__(
            self,
            architecture: RandomForestArchitecture,
            training_config: TrainingConfig,
            encoder: MoleculeEncoder,
            static_embeddings: bool = True,
            random_seed: int = 42,
            **kwargs
    ):
        """
        Args:
            architecture: RandomForestArchitecture instance containing model hyperparameters.
            training_config: TrainingConfig instance containing training hyperparameters.
            encoder: MoleculeEncoder instance for encoding molecular inputs.
            static_embeddings: If True, precompute and store embeddings for training data.
            random_seed: Random seed for reproducibility.
        """
        if not static_embeddings:
            raise ValueError("Random Forest model only supports static embeddings.")

        Model.__init__(self, architecture, training_config, encoder, static_embeddings, random_seed)

        self.model = RandomForestRegressor(
            n_estimators=architecture.n_estimators,
            max_depth=architecture.max_depth,
            min_samples_split=architecture.min_samples_split,
            max_features=architecture.max_features,
            n_jobs=training_config.n_jobs,
            random_state=random_seed
        )

    def train_model(
            self,
            train_smiles: np.ndarray,
            train_labels: torch.Tensor,
            train_embeddings: torch.Tensor = None
    ):
        """
        Trains the Random Forest model on the provided data.

        Args:
            train_smiles (np.ndarray): Array of SMILES strings for training data.
            train_labels (torch.Tensor): Tensor of target labels for training data.
            train_embeddings (torch.Tensor, optional): Tensor of precomputed embeddings for training data. Only used if
                                                       static_embeddings is True.
        """
        if train_embeddings is None:
            train_embeddings = self.encoder.encode(train_smiles)

        self.model.fit(train_embeddings.numpy(), train_labels.numpy().ravel())

    def retrain_model(
            self,
            train_smiles: np.ndarray,
            train_labels: torch.Tensor,
            train_embeddings: torch.Tensor = None
    ):
        """
        Retrains the Random Forest model on the provided data. Since Random Forest does not support incremental training,
        this method simply retrains the model from scratch.

        Args:
            train_smiles (np.ndarray): Array of SMILES strings for training data.
            train_labels (torch.Tensor): Tensor of target labels for training data.
            train_embeddings (torch.Tensor, optional): Tensor of precomputed embeddings for training data. Only used if
                                                       static_embeddings is True.
        """
        self.train_model(train_smiles, train_labels, train_embeddings)

    def predict(self, embeddings: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Makes predictions using the trained Random Forest model.

        Args:
            embeddings (torch.Tensor, optional): Tensor of precomputed embeddings for prediction data. If None,
                                                 embeddings will be computed using the encoder.

        Returns:
            torch.Tensor: Predicted means of shape (N, 1).
            torch.Tensor: Predicted variances of shape (N, 1).

        """
        if embeddings is None:
            raise ValueError("embeddings must be provided for prediction in Random Forest model.")

        # Get predictions from each tree in the forest
        all_tree_predictions = np.array([tree.predict(embeddings.numpy()) for tree in self.model.estimators_])
        means = np.mean(all_tree_predictions, axis=0)
        variances = np.var(all_tree_predictions, axis=0)

        return (
            torch.tensor(means, dtype=torch.get_default_dtype()).unsqueeze(-1),
            torch.tensor(variances, dtype=torch.get_default_dtype()).unsqueeze(-1)
        )

