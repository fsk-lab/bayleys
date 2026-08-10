from typing import Optional
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from laplace import Laplace, ParametricLaplace

from .base import TrainingConfig

from ..encoders.base import MoleculeEncoder
from .nn import NeuralNetworkArchitecture, NeuralNetworkModel
from ..config import get_config


class LaplaceNeuralNetworkModel(NeuralNetworkModel):
    """
    Neural Network surrogate model that uses a post-hoc Laplace approximation to obtain uncertainty estimates. The model
    is trained in a standard way and the Laplace approximation is applied after training to obtain a Bayesian Neural
    Network surrogate. The Laplace approximation is applied to the last layer of the neural network.
    """
    name = f"LaplaceNN"

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
        NeuralNetworkModel.__init__(self, architecture, training_config, encoder, static_embeddings, random_seed)
        self.bnn: Optional[ParametricLaplace] = None

    def train_model(
            self,
            train_smiles: np.ndarray,
            train_labels: torch.Tensor,
            train_embeddings: torch.Tensor = None
    ):
        """
        Trains the neural network model on the provided data. Uses a post-hoc Laplace strategy.

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
            train_idx, val_idx = all_indices[: int(0.8 * len(all_indices))], all_indices[int(0.8 * len(all_indices)) :]
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

        del self.bnn
        self.bnn = Laplace(
            self.readout,
            likelihood="regression",
            subset_of_weights="last_layer",
            hessian_structure="kron",
        )
        self.bnn.fit(train_loader)
        self.bnn.optimize_prior_precision(pred_type="glm", method="marglik")

    @torch.inference_mode()
    def predict(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts the target labels and the corresponding variances using the trained Laplace-approximated Bayesian
        Neural Network.

        Args:
            embeddings (torch.Tensor, optional): Tensor of precomputed embeddings for prediction data.

        Returns:
            torch.Tensor: Predicted means of shape (N, 1).
            torch.Tensor: Predicted variances of shape (N, 1).
        """
        if self.bnn is None:
            raise ValueError("Model must be trained before prediction.")

        device = get_config().device
        self.to(device)

        embeddings = embeddings.to(device)
        self.eval()
        means, variances = self.bnn(embeddings)

        return means.cpu(), variances.squeeze(-1).cpu()
