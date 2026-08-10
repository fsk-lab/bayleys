import importlib
import numpy as np
import torch

from .parallelizable import ParallelMoleculeEncoder


class PicklableRDKitDescriptorCalculator(object):
    """
    Picklable wrapper for RDKit descriptor calculators. This is necessary for multiprocessing. Wraps all the RDKit
    imports and function calls in a single class which can be pickled and passed to joblib.Parallel.
    """
    def __init__(self, **kwargs):
        self._chem = importlib.import_module('rdkit.Chem')

    def __call__(self, smiles: np.ndarray) -> np.ndarray:
        """
        Calculates RDKit descriptors for a batch of SMILES strings.
        """
        descs = importlib.import_module('rdkit.Chem.Descriptors')
        descriptor_funcs = [func for name, func in getattr(descs, "_descList")]

        descriptors = np.zeros((len(smiles), len(descriptor_funcs)), dtype=np.float64)
        for i, smi in enumerate(smiles):
            mol = getattr(self._chem, "MolFromSmiles")(smi)
            for j, func in enumerate(descriptor_funcs):
                try:
                    descriptors[i, j] = func(mol)
                except Exception as e:
                    descriptors[i, j] = np.nan

        return descriptors


class RDKitDescriptorEncoder(ParallelMoleculeEncoder):
    """
    Encoder for generating simple, graph-level molecular descriptors using the RDKit library. Encodes each molecule as
    a vector of molecular descriptors.
    """

    learnable = False
    embedding_dim = None  # Can only be obtained after first encoding run.

    def __init__(self, n_jobs: int = -2, batch_size: int = 1000, **kwargs):
        """
        Args:
            n_jobs: Number of processors encoding should be parallelized. Defaults to -2 (all but one core).
            batch_size: Number of molecules to encode per batch.
        """
        super().__init__(n_jobs=n_jobs, batch_size=batch_size)

        self.picklable_encoder = PicklableRDKitDescriptorCalculator()

    def encode(self, smiles: np.ndarray) -> torch.Tensor:
        """
        Encodes a list of SMILES strings into RDKit descriptors using parallel processing. Uses the method of the
        parent class, removes all NaN and redundant features, and scales the features to [0, 1] using Min-Max scaling.

        Args:
            smiles: Array of SMILES strings to encode.

        Returns:
            Tensor: Encoded representation of the input SMILES strings.
        """
        embeddings = super().encode(smiles)

        # Remove NaN features and redundant features
        contains_nans = torch.isnan(embeddings).any(dim=0)
        embeddings = embeddings[:, ~contains_nans]

        # Remove zero-variance features
        zero_variance_mask = torch.isclose(torch.var(embeddings, dim=0), torch.tensor(0.0))
        embeddings = embeddings[:, ~zero_variance_mask]

        # Min-Max Scale to [0, 1]
        min_vals = torch.min(embeddings, dim=0).values
        max_vals = torch.max(embeddings, dim=0).values
        embeddings = (embeddings - min_vals) / (max_vals - min_vals)

        # Set embedding dimension
        self.embedding_dim = embeddings.shape[1]

        return embeddings

    @property
    def name(self) -> str:
        return "RDKitDescriptors"
