import importlib
import numpy as np
import torch

from .parallelizable import ParallelMoleculeEncoder


class PicklableMordredDescriptorCalculator(object):
    """
    Picklable wrapper for Mordred featurizers. This is necessary for multiprocessing. Wraps all the RDKit imports and
    function calls in a single class which can be pickled and passed to joblib.Parallel.
    """
    def __init__(self, **kwargs):
        self._chem = importlib.import_module('rdkit.Chem')
        mordred = importlib.import_module('mordred')
        descriptors = importlib.import_module('mordred.descriptors')
        self._calculator = getattr(mordred, "Calculator")(descriptors, ignore_3D=True)

    def __call__(self, smiles: np.ndarray) -> np.ndarray:
        """
        Calculates Mordred descriptors for a batch of SMILES strings.
        """
        descriptors = np.zeros((len(smiles), len(self._calculator.descriptors)), dtype=np.float32)
        for i, smi in enumerate(smiles):
            mol = getattr(self._chem, "MolFromSmiles")(smi)
            descriptors[i, :] = np.asarray(self._calculator(mol)).astype(np.float32)

        return descriptors


class MordredEncoder(ParallelMoleculeEncoder):
    """
    Encoder for generating simple, graph-level molecular descriptors using the RDKit and Mordred libraries. Encodes
    molecules as a vector of molecular descriptors.
    """

    learnable = False
    embedding_dim = None  # Can only be obtained after first encoding run.

    def __init__(self, n_jobs: int = -2, batch_size: int = 1000, **kwargs):
        """
        Args:
            n_jobs: Number of processors encoding should be parallelized. Defaults to -2 (all but one core).
            batch_size: Number of molecules to encode per batch.
            **kwargs: Further keyword arguments to be passed to the encoder.
        """
        super().__init__(n_jobs=n_jobs, batch_size=batch_size)

        self.picklable_encoder = PicklableMordredDescriptorCalculator(**kwargs)

    def encode(self, smiles: np.ndarray) -> torch.Tensor:
        """
        Encodes a list of SMILES strings into Mordred descriptors using parallel processing. Uses the method of the
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
        return "Mordred"
