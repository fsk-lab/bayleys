from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .base import ModelArchitecture, TrainingConfig, Model

from ..encoders.base import MoleculeEncoder
from ..utils.torch import reset_weights
from ..config import get_config


ACTIVATION_FACTORY = {"ReLU": torch.nn.ReLU, "Tanh": torch.nn.Tanh, "Sigmoid": torch.nn.Sigmoid}


@dataclass
class NeuralNetworkArchitecture(ModelArchitecture):
    """
    Dataclass that holds architecture parameters for neural network surrogate models.
    """
    num_hidden_layers: int = 2
    hidden_dim: int | list[int] = 50
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

    def get_activation_function(self):
        """
        Returns the activation function based on the architecture configuration.

        Returns:
            The specified activation function.
        """
        return ACTIVATION_FACTORY[self.activation_function]()

    def get_fcnn_module(self, input_dim: int) -> torch.nn.Module:
        """
        Constructs and returns a fully connected neural network module based on the architecture configuration.

        Args:
            input_dim (int): Dimensionality of the input features.

        Returns:
            torch.nn.Module: The constructed fully connected neural network module.
        """
        if isinstance(self.hidden_dim, int):
            hidden_dims = [self.hidden_dim] * self.num_hidden_layers
        elif isinstance(self.hidden_dim, list) and len(self.hidden_dim) == self.num_hidden_layers:
            hidden_dims = self.hidden_dim
        else:
            raise ValueError("hidden_dim must be either an int or a list of ints with a length of num_hidden_layers.")

        layers = []
        current_dim = input_dim

        for i in range(self.num_hidden_layers):
            layers.append(torch.nn.Linear(current_dim, hidden_dims[i]))
            layers.append(self.get_activation_function())
            layers.append(torch.nn.Dropout(self.dropout_rate))
            current_dim = hidden_dims[i]

        layers.append(torch.nn.Linear(current_dim, 1))

        return torch.nn.Sequential(*layers)


class NeuralNetworkModel(Model, torch.nn.Module):
    """
    Neural Network surrogate model for molecular property prediction.
    """
    name = f"NN"

    default_architecture = NeuralNetworkArchitecture()
    default_training_config = TrainingConfig()

    def __init__(
            self,
            architecture: NeuralNetworkArchitecture,
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
        self.readout = architecture.get_fcnn_module(input_dim=encoder.embedding_dim)
        self.rng = torch.Generator()
        self.rng.manual_seed(random_seed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Neural Network model.

        Args:
            x (torch.Tensor): Input tensor of shape (n_samples, n_features).

        Returns:
            torch.Tensor: Output tensor after passing through the neural network (shape: (n_samples, 1)).
        """
        return self.readout(x)

    def train_model(
            self,
            train_smiles: np.ndarray,
            train_labels: torch.Tensor,
            train_embeddings: torch.Tensor = None
    ):
        """
        Trains the neural network model on the provided data.

        Args:
            train_smiles (np.ndarray): Array of SMILES strings for training data.
            train_labels (torch.Tensor): Tensor of target labels for training data.
            train_embeddings (torch.Tensor, optional): Tensor of precomputed embeddings for training data. Only used if
                                                       static_embeddings is True.
        """
        if self.static_embeddings is True:
            if train_embeddings is None:
                raise ValueError("train_embeddings must be provided when static_embeddings is True.")
        else:
            raise NotImplementedError("On-the-fly embedding adaptation is not implemented yet.")

        device = get_config().device
        self.to(device)

        optimizer = self.training_config.get_optimizer(self.readout.parameters())
        loss = torch.nn.MSELoss()

        do_early_stopping = self.training_config.early_stopping and train_smiles.shape[0] >= 10

        if do_early_stopping:
            all_indices = torch.randperm(train_smiles.shape[0], generator=self.rng)
            train_idx, val_idx = all_indices[: int(0.8 * len(all_indices))], all_indices[int(0.8 * len(all_indices)):]
            train_smiles, val_smiles = train_smiles[train_idx.numpy()], train_smiles[val_idx.numpy()]
            train_labels, val_labels = train_labels[train_idx], train_labels[val_idx]
            train_embeddings, val_embeddings = train_embeddings[train_idx], train_embeddings[val_idx]
            best_val_loss, best_model, epochs_no_improve = torch.inf, None, 0

        train_loader = DataLoader(
            dataset=TensorDataset(train_embeddings, train_labels),
            batch_size=self.training_config.batch_size,
            shuffle=True,
            generator=self.rng
        )

        self.readout.train()
        for epoch in range(self.training_config.num_epochs):

            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                predictions = self.forward(batch_x)
                batch_loss = loss(predictions, batch_y)
                batch_loss.backward()
                optimizer.step()

            if do_early_stopping:
                val_loader = DataLoader(
                    dataset=TensorDataset(val_embeddings, val_labels),
                    batch_size=self.training_config.batch_size,
                    shuffle=False,
                )

                val_loss = 0.0
                self.readout.eval()
                with torch.inference_mode():
                    for val_x, val_y in val_loader:
                        val_x, val_y = val_x.to(device), val_y.to(device)
                        val_predictions = self.forward(val_x)
                        val_loss += loss(val_predictions, val_y).item()

                val_loss /= len(val_loader)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = self.readout.state_dict()
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= self.training_config.early_stopping_patience:
                    if best_model is None:
                        epochs_no_improve = 0
                    else:
                        self.readout.load_state_dict(best_model)
                        break

    def retrain_model(
            self,
            train_smiles: np.ndarray,
            train_labels: torch.Tensor,
            train_embeddings: torch.Tensor = None
    ):
        """
        Retrains the neural network model on the provided data. For any neural network model, retraining is equivalent
        to resetting all model parameters and training from scratch.

        Args:
            train_smiles (np.ndarray): Array of SMILES strings for training data.
            train_labels (torch.Tensor): Tensor of target labels for training data.
            train_embeddings (torch.Tensor, optional): Tensor of precomputed embeddings for training data. Only used if
                                                       static_embeddings is True.
        """
        self.readout.apply(reset_weights)
        self.train_model(train_smiles, train_labels, train_embeddings)

    @torch.inference_mode()
    def predict(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts the target labels for the embeddings using the neural network model. Since a deterministic
        neural network does not provide uncertainty estimates, the variance is returned as zeros.

        Args:
            embeddings (torch.Tensor, optional): Tensor of precomputed embeddings for prediction data.

        Returns:
            torch.Tensor: Predicted means of shape (N, 1).
            torch.Tensor: Predicted variances of shape (N, 1).
        """
        device = get_config().device
        self.to(device)

        embeddings = embeddings.to(device)
        self.readout.eval()
        predictions = self.forward(embeddings).cpu()

        return predictions, torch.zeros_like(predictions)

