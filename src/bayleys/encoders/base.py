from abc import ABC, abstractmethod
import numpy as np

from ..utils import TensorLike


class MoleculeEncoder(ABC):
    """
    Abstract base class for molecule encoders.

    Attributes:
        embedding_dim (int): Dimension of the encoded embeddings.
        learnable (bool): Indicates whether the encoder uses learnable embeddings (e.g., neural network-based)
                          or fixed descriptors (e.g., physicochemical properties).
    """
    embedding_dim: int
    learnable: bool = False

    @abstractmethod
    def encode(self, smiles: np.ndarray) -> TensorLike:
        """
        Encodes a list of SMILES strings into a desired representation.

        Args:
            smiles: Array of SMILES strings to encode.

        Returns:
            TensorLike: Encoded representation of the input SMILES strings.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Returns the name of the encoder.

        Returns:
            str: Name of the encoder.
        """
        raise NotImplementedError