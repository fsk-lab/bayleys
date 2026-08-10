import time
from logging import getLogger
import torch

from .config import ModelConfig, MODEL_FACTORY

from ..encoders import MoleculeEncoderConfig, LLMMoleculeEncoder
from ..molecule_library import MoleculeLibrary
from ..utils import StandardScaler


class SurrogateModel(object):
    """
    Wrapper class for Bayesian surrogate models for molecular property prediction. Integrates molecule encoding,
    surrogate model training, and inference functionalities.
    """

    def __init__(
            self,
            encoder_config: MoleculeEncoderConfig,
            model_config: ModelConfig,
    ):
        """

        Args:
            encoder_config (MoleculeEncoderConfig): Configuration for the molecule encoder.
            model_config (ModelConfig): Configuration for the surrogate model.
        """
        self.encoder_config = encoder_config
        self.model_config = model_config
        self.encoder = encoder_config.get_encoder()
        self.model = None
        self.y_scaler = StandardScaler()

        self.logger = getLogger("bayleys")

    def compute_embeddings(self, library: MoleculeLibrary) -> torch.Tensor:
        """
        Generates embeddings for the molecules in the provided library using the configured encoder.

        Args:
            library (MoleculeLibrary): MoleculeLibrary object containing the molecules to be encoded.

        Returns:
            torch.Tensor: Tensor of shape (N, D) containing the generated embeddings.
        """
        start_time = time.time()

        if isinstance(self.encoder, LLMMoleculeEncoder):
            self.encoder.max_length = library.max_smiles_length

        embeddings = self.encoder.encode(library.smiles)

        self.logger.debug(f"{self.__class__.__name__}: Generated embeddings ({embeddings.shape[1]} dimensions) "
                          f"for {len(library)} molecules in {time.time() - start_time:.1f} seconds.")

        return embeddings

    def train_model(self, train_library: MoleculeLibrary):
        """
        Trains the surrogate model using the provided training molecule library.

        Args:
            train_library (MoleculeLibrary): MoleculeLibrary object containing the training data. Must contain the
                                             `label` attribute with target property values.
        """
        train_y = self.y_scaler.fit_transform(train_library.labels)

        start_time = time.time()
        if self.model is None:
            self.model = self.model_config.get_model(self.encoder)
            self.model.train_model(
                train_smiles=train_library.smiles,
                train_embeddings=train_library.embeddings,
                train_labels=train_y,
            )
        else:
            self.model.retrain_model(
                train_smiles=train_library.smiles,
                train_embeddings=train_library.embeddings,
                train_labels=train_y,
            )

        self.logger.debug(f"{self.__class__.__name__}: Surrogate model training ({self.model.__class__.__name__}) "
                          f"with {len(train_library)} data points completed in {time.time() - start_time:.1f} seconds.")

    @torch.inference_mode()
    def predict(self, test_library: MoleculeLibrary) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs inference using the surrogate model on the provided test features. Does not use a DataLoader, but
        performs "manual" batching, which is empirically much faster than using a DataLoader.

        Args:
            test_library (MoleculeLibrary): MoleculeLibrary object containing the test data.

        Returns:
            torch.Tensor: Tensor of shape (M, 1) containing the predicted means.
            torch.Tensor: Tensor of shape (M, 1) containing the predicted variances.
            torch.Tensor: Tensor of shape (M, D) containing the embeddings used for prediction.
        """
        means, vars, embeddings = [], [], []

        start_time = time.time()

        for idx in range(0, len(test_library), self.model_config.inference_batch_size):
            batch_smiles = test_library.smiles[idx: idx + self.model_config.inference_batch_size]
            if self.model_config.static_embeddings is True:
                batch_embeddings = test_library.embeddings[idx: idx + self.model_config.inference_batch_size]
            else:
                batch_embeddings = self.encoder.encode(batch_smiles)

            batch_means, batch_vars = self.model.predict(batch_embeddings)
            means.append(batch_means), vars.append(batch_vars)
            embeddings.append(batch_embeddings)

        means = self.y_scaler.inverse_transform(torch.cat(means, dim=0))
        vars = self.y_scaler.inverse_transform_variance(torch.cat(vars, dim=0))
        embeddings = torch.cat(embeddings, dim=0)

        self.logger.debug(f"{self.__class__.__name__}: Surrogate model inference ({self.model.__class__.__name__}) "
                          f"on {len(test_library)} data points completed in {time.time() - start_time:.1f} seconds.")

        return means.to(torch.get_default_dtype()), vars.to(torch.get_default_dtype()), embeddings

    @property
    def name(self) -> str:
        """
        Returns the name of the surrogate model class.

        Returns:
            str: Name of the surrogate model class.
        """
        model_name = MODEL_FACTORY[self.model_config.model_type].name
        return f"{self.encoder.name}_{model_name}"

