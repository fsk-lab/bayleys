from dataclasses import dataclass
import numpy as np
import torch

from .base import ModelArchitecture, TrainingConfig, Model
from .nn import ACTIVATION_FACTORY, NeuralNetworkArchitecture, NeuralNetworkModel

from ..encoders.base import MoleculeEncoder


@dataclass
class NeuralNetworkEnsembleArchitecture(ModelArchitecture):
    """
    Dataclass that holds architecture parameters for neural network ensembles.
    """
    num_networks: int = 10
    num_hidden_layers: int = 2
    hidden_dim: int = 50
    dropout_rate: float = 0.1
    activation_function: str = "ReLU"

    def __post_init__(self):
        """
        Validates the neural network architecture configuration after initialization by performing the following
        checks:
          - Ensures that the specified activation function exists in the ACTIVATION_FACTORY.
        """
        if self.activation_function not in ACTIVATION_FACTORY:
            raise ValueError(f"Activation function '{self.activation_function}' is not recognized!")

    @property
    def nn_architecture(self) -> NeuralNetworkArchitecture:
        """
        Returns a NeuralNetworkArchitecture instance with the same architecture parameters as the ensemble.

        Returns:
            NeuralNetworkArchitecture: The corresponding neural network architecture.
        """
        return NeuralNetworkArchitecture(
            num_hidden_layers=self.num_hidden_layers,
            hidden_dim=self.hidden_dim,
            dropout_rate=self.dropout_rate,
            activation_function=self.activation_function,
        )


class NeuralNetworkEnsembleModel(Model, torch.nn.Module):
    """
    Neural Network surrogate model for molecular property prediction.
    """
    name = "NNEns"

    default_architecture = NeuralNetworkEnsembleArchitecture()
    default_training_config = TrainingConfig()

    def __init__(
            self,
            architecture: NeuralNetworkEnsembleArchitecture,
            training_config: TrainingConfig,
            encoder: MoleculeEncoder,
            static_embeddings: bool = True,
            random_seed: int = 42,
    ):
        """
        Initializes the Neural Network surrogate model.

        Args:
            architecture (NeuralNetworkArchitecture): Architecture configuration for the neural network.
            training_config: Training configuration for the neural network.
            encoder: Molecule encoder to be used for feature extraction.
            static_embeddings (bool): Whether to use static embeddings from the encoder.
            random_seed (int): Random seed for reproducibility.
        """
        torch.nn.Module.__init__(self)
        Model.__init__(self, architecture, training_config, encoder, static_embeddings, random_seed)
        self.networks = [
            NeuralNetworkModel(
                architecture=architecture.nn_architecture,
                training_config=training_config,
                encoder=encoder,
                static_embeddings=static_embeddings,
                random_seed=random_seed + i
            )
            for i in range(architecture.num_networks)
        ]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Forward pass through all neural networks in the ensemble.

        Args:
            x (torch.Tensor): Input tensor of shape (n_samples, n_features).

        Returns:
            list[torch.Tensor]: List of output tensors from each neural network in the ensemble.
        """
        return [network(x) for network in self.networks]  # List of (n_samples, 1) tensors

    def train_model(
            self,
            train_smiles: np.ndarray,
            train_labels: torch.Tensor,
            train_embeddings: torch.Tensor = None
    ):
        """
        Trains the neural network ensemble on the provided data.

        Args:
            train_smiles (np.ndarray): Array of SMILES strings for training data.
            train_labels (torch.Tensor): Tensor of target labels for training data.
            train_embeddings (torch.Tensor, optional): Tensor of precomputed embeddings for training data. Only used if
                                                       static_embeddings is True.
        """
        for model in self.networks:
            model.train_model(train_smiles, train_labels, train_embeddings)

    def retrain_model(
            self,
            train_smiles: np.ndarray,
            train_labels: torch.Tensor,
            train_embeddings: torch.Tensor = None
    ):
        """
        Retrains the neural network ensemble on the provided data.

        Args:
            train_smiles (np.ndarray): Array of SMILES strings for training data.
            train_labels (torch.Tensor): Tensor of target labels for training data.
            train_embeddings (torch.Tensor, optional): Tensor of precomputed embeddings for training data. Only used if
                                                       static_embeddings is True.
        """
        for model in self.networks:
            model.retrain_model(train_smiles, train_labels, train_embeddings)

    @torch.inference_mode()
    def predict(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts the target labels for the embeddings using the neural network ensemble. The mean and variance
        of the predictions from all networks in the ensemble are returned.

        Args:
            embeddings (torch.Tensor, optional): Tensor of precomputed embeddings for prediction data.

        Returns:
            torch.Tensor: Predicted means of shape (N, 1).
            torch.Tensor: Predicted variances of shape (N, 1).
        """
        predictions = []
        for model in self.networks:
            preds, _ = model.predict(embeddings)
            predictions.append(preds)  # (N, 1)
        predictions = torch.stack(predictions, dim=0)

        return predictions.mean(dim=0), predictions.var(dim=0)

