import importlib
import numpy as np

from .parallelizable import ParallelMoleculeEncoder


class PicklableMorganFingerprintCalculator(object):
    """
    Picklable wrapper for calculating defined fingerprints of molecules using RDKit. This can be used for accelerated
    fingerprint calculation using joblib's Parallel module.

    Calculates a Morgan fingerprint with specified radius and length.
    """

    def __init__(self, radius: int, n_bits: int):

        self.rdkit_chem = importlib.import_module('rdkit.Chem')
        self.rdkit_allchem = importlib.import_module('rdkit.Chem.AllChem')
        self.radius = radius
        self.n_bits = n_bits

    def __call__(self, smiles: np.ndarray) -> np.ndarray:
        """
        Calculates Morgan fingerprints for a batch of SMILES strings.
        """
        fp_generator = getattr(self.rdkit_chem, "rdFingerprintGenerator").GetMorganGenerator(
            radius=self.radius,
            fpSize=self.n_bits
        )
        fingerprints = np.zeros((smiles.shape[0], self.n_bits))
        for i, smi in enumerate(smiles):
            mol = getattr(self.rdkit_chem, "MolFromSmiles")(smi)
            fingerprints[i, :] = fp_generator.GetFingerprint(mol)

        return fingerprints


class MorganFingerprintEncoder(ParallelMoleculeEncoder):
    """
    Encoder for Morgan fingerprints using RDKit, following the general encoder API. Encodes molecules as Morgan fingerprint
    bit vectors.
    """

    learnable = False

    def __init__(self, radius: int = 2, n_bits: int = 1024, n_jobs: int = -2, batch_size: int = 1000, **kwargs):
        """
        Args:
            radius: Radius of the Morgan fingerprint.
            n_bits: Length of the fingerprint in bits.
            n_jobs: Number of processors encoding should be parallelized. Defaults to -2 (all but one core).
            batch_size: Number of molecules to encode per batch.
        """
        super().__init__(n_jobs=n_jobs, batch_size=batch_size)

        self.picklable_encoder = PicklableMorganFingerprintCalculator(radius, n_bits)
        self.embedding_dim = n_bits

    @property
    def name(self) -> str:
        return f"MorganFingerprint-{self.picklable_encoder.radius}-{self.picklable_encoder.n_bits}"
