import time
from abc import ABC
from logging import getLogger
import numpy as np
import torch
from joblib import Parallel, delayed, cpu_count

from .base import MoleculeEncoder


class ParallelMoleculeEncoder(MoleculeEncoder, ABC):
    """
    Abstract base class for parallelizable molecule encoders, i.e. molecule encoders that can encode batches of
    molecules using joblib's Parallel module.

    All sublcasses must use a picklable encoder that can encode a batch of SMILES strings when called.
    """

    def __init__(self, n_jobs: int = -2, batch_size: int = 1000, **kwargs):
        """
        Args:
            n_jobs: Number of processors encoding should be parallelized. Defaults to -2 (all but one core).
            **kwargs: Further keyword arguments for the specific subclasses.
        """
        super().__init__(**kwargs)

        self.picklable_encoder = None  # To be defined in subclasses
        self.batch_size = batch_size
        self.n_jobs = n_jobs if n_jobs > 0 else cpu_count() + n_jobs

        self.logger = getLogger("bayleys")

    def encode(self, smiles: np.ndarray) -> torch.Tensor:
        """
        Encodes a list of SMILES strings into a desired representation using parallel processing.

        Args:
            smiles: Array of SMILES strings to encode.

        Returns:
            Tensor: Encoded representation of the input SMILES strings.
        """
        self.logger.debug(f"{self.__class__.__name__}: Starting to encode {len(smiles)} molecules using {self.n_jobs} "
                          f"parallel jobs...")
        start_time = time.time()

        embeddings = Parallel(n_jobs=self.n_jobs)(
            delayed(self.picklable_encoder)(smiles[i: i + self.batch_size])
            for i in range(0, len(smiles), self.batch_size)
        )
        embeddings = np.vstack(embeddings)

        self.logger.debug(f"{self.__class__.__name__}: Encoding completed in {time.time() - start_time:.1f} seconds.")

        return torch.Tensor(embeddings)
