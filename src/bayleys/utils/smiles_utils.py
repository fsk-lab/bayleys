from pathlib import Path
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


_POPCOUNT8 = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)
_UINT64_MIX = np.uint64(0x9E3779B97F4A7C15)


def canonicalize(smiles: str) -> str:
    """
    Canonicalize a SMILES string.

    Args:
        smiles (str): The input SMILES string.

    Returns:
        str: The canonicalized SMILES string (or the original string if it cannot be parsed by the RDKit).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    return Chem.MolToSmiles(mol, canonical=True)


def randomize_smiles(smiles: str) -> str:
    """
    Randomize a SMILES string.

    Args:
        smiles (str): The input SMILES string.

    Returns:
        str: A randomized SMILES string (or the original string if it cannot be parsed by the RDKit).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    return Chem.MolToSmiles(mol, doRandom=True, canonical=False)


def randomize_smiles_generator(smiles: str, n_random: int) -> str:
    """
    Generator that yields randomized SMILES strings for a given input SMILES string. First yields the canonical SMILES,
    then yields n_random - 1 randomized SMILES strings.

    Args:
        smiles (str): The input SMILES string.
        n_random (int): The number of randomized SMILES strings to generate.

    Yields:
        str: A randomized SMILES string (or the original string if it cannot be parsed by the RDKit).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        for _ in range(n_random):
            yield smiles
    else:
        yield Chem.MolToSmiles(mol, canonical=True)
        for _ in range(n_random - 1):
            yield Chem.MolToSmiles(mol, doRandom=True, canonical=False)


def canonicalize_smiles_batch(smiles_batch: list | np.ndarray) -> list | np.ndarray:
    """
    Canonicalize a batch of SMILES strings.

    Args:
        smiles_batch (list | np.ndarray): A batch of SMILES strings as a list or numpy array.

    Returns:
        list | np.ndarray: A batch of canonicalized SMILES strings in the same format as the input.
    """
    if isinstance(smiles_batch, np.ndarray):
        return np.array([canonicalize(smiles) for smiles in smiles_batch])
    else:
        return [canonicalize(smiles) for smiles in smiles_batch]


def randomize_smiles_batch(smiles_batch: list | np.ndarray) -> list | np.ndarray:
    """
    Randomize a batch of SMILES strings.

    Args:
        smiles_batch (list | np.ndarray): A batch of SMILES strings as a list or numpy array.

    Returns:
        list | np.ndarray: A batch of randomized SMILES strings in the same format as the input.
    """
    if isinstance(smiles_batch, np.ndarray):
        return np.array([randomize_smiles(smiles) for smiles in smiles_batch])
    else:
        return [randomize_smiles(smiles) for smiles in smiles_batch]


def get_minhash_signature(on_bits: np.ndarray, minhash_table: np.ndarray) -> np.ndarray:
    """
    Compute the MinHash signature for a given set of on bits in a fingerprint.

    Args:
        on_bits (np.ndarray): An array of indices of the on bits in the fingerprint.
        minhash_table (np.ndarray): A precomputed table of random hash values for each bit index.

    Returns:
        np.ndarray: The MinHash signature as an array of uint32 values.
    """
    if on_bits.size == 0:
        return np.zeros(minhash_table.shape[0], dtype=np.uint32)
    return minhash_table[:, on_bits].min(axis=1)


def get_minhash_table(num_permutations: int, fp_size: int, seed: int = 42) -> np.ndarray:
    """
    Generate a MinHash table of random hash values for each bit index in the fingerprint.

    Args:
        num_permutations (int): The number of permutations (i.e., the size of the MinHash signature).
        fp_size (int): The size of the fingerprint (i.e., the number of bits).
        seed (int): A random seed for reproducibility.

    Returns:
        np.ndarray: A 2D array of shape (num_permutations, fp_size) containing random uint32 hash values.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(low=0, high=np.iinfo(np.uint32).max, size=(num_permutations, fp_size), dtype=np.uint32)


def hash_lsh_band(signatures: np.ndarray, band_idx: int, band_size: int) -> np.ndarray:
    """
    Compute the hash keys for a given band of the MinHash signatures for LSH candidate generation. The hash keys are
    computed using a simple mixing function that combines the values in the band of the signature for each SMILES
    string.

    Args:
        signatures (np.ndarray): An array of shape (num_smiles, num_permutations).
        band_idx (int): The index of the band to compute the hash keys for.
        band_size (int): The number of rows in each band.

    Returns:
        np.ndarray: An array of shape (num_smiles,) containing the hash keys for the specified band, as uint64 values.
    """
    keys = np.full(signatures.shape[0], np.uint64(1469598103934665603), dtype=np.uint64)
    for col in range(band_size * band_idx, band_size * (band_idx + 1)):
        values = signatures[:, col].astype(np.uint64, copy=False)
        keys ^= values + _UINT64_MIX + (keys << np.uint64(6)) + (keys >> np.uint64(2))
    return keys


def row_popcount_uint64(arr: np.ndarray) -> np.ndarray:
    """
    Compute the population count (i.e., the number of set bits) for each row in an array of uint64 values.

    Args:
        arr (np.ndarray): A 1D or 2D array of uint64 values.

    Returns:
        np.ndarray: A 1D array containing the population count for each row.
    """
    arr = np.asarray(arr, dtype=np.uint64)

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    if arr.shape[0] == 0:
        return np.zeros(0, dtype=np.uint16)

    byte_view = np.ascontiguousarray(arr).view(np.uint8).reshape(arr.shape[0], -1)
    return _POPCOUNT8[byte_view].sum(axis=1, dtype=np.uint16)


def batch_tanimoto_similarity(
        all_fps: np.ndarray,
        all_bit_counts: np.ndarray,
        anchor_idx: int,
        candidate_indices: np.ndarray
) -> np.ndarray:
    """
    Compute the Tanimoto similarity between an anchor fingerprint and a set of candidate fingerprints.

    Args:
        all_fps (np.ndarray): An array of shape (num_smiles, num_words) containing the fingerprints.
        all_bit_counts (np.ndarray): An array of shape (num_smiles,) containing the bit counts for each fingerprint.
        anchor_idx (int): The index of the anchor fingerprint.
        candidate_indices (np.ndarray): An array of indices for the candidate fingerprints.
    """
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)

    if candidate_indices.size == 0:
        return np.zeros(0, dtype=np.float32)

    anchor_fp = all_fps[anchor_idx]
    candidate_fps = all_fps[candidate_indices]

    intersection = row_popcount_uint64(np.bitwise_and(anchor_fp, candidate_fps)).astype(np.uint32)

    denominator = (
        np.uint32(all_bit_counts[anchor_idx])
        + all_bit_counts[candidate_indices].astype(np.uint32)
        - intersection
    )

    similarities = np.zeros(candidate_indices.shape[0], dtype=np.float32)
    mask = denominator > 0
    similarities[mask] = intersection[mask] / denominator[mask]

    return similarities


class SMILESPairSampler:
    """
    Class to sample positive (highly similar) and negative (highly dissimilar) examples of SMILES strings for each
    candidate from a given list of SMILES strings using fingerprint-based similarity and locality-sensitive hashing
    (LSH).

    Args:
        all_smiles (np.ndarray): An array of SMILES strings to sample from.
        cache_dir (Path): A directory to cache fingerprint similarities and other intermediate results to enable
                          memory-efficient sampling.
        fp_size (int): The size of the Morgan fingerprint in bits (default: 2048).
        fp_radius (int): The radius parameter for the Morgan fingerprint (default: 2).
        batch_size (int): The number of SMILES strings to process in each batch when computing fingerprints and
                          similarities (default: 50,000).
        num_permutations (int): The number of permutations to use for the MinHash signatures (default: 64).
        band_size (int): The number of rows to use in each band for LSH candidate generation (default: 4).
        max_bucket_size (int): The maximum number of candidates to consider in each LSH bucket when sampling positive
                               examples (default: 100).
        max_candidates_per_anchor_bucket (int): The maximum number of candidates to consider for each anchor
                                                from each LSH bucket when sampling positive examples (default: 100).
        max_negative_batches (int): The maximum number of random batches to sample when looking for negative examples
                                    for each anchor (default: 10).
        similarity_threshold (float): The minimum Tanimoto similarity required for a candidate to be considered a
                                      positive example (default: 0.7).
        dissimilarity_threshold (float): The maximum Tanimoto similarity allowed for a candidate to be considered a
                                         negative example (default: 0.1).
        seed (int): A random seed for reproducibility (default: 42).

    """
    def __init__(
            self,
            all_smiles: np.ndarray,
            cache_dir: Path,
            fp_size: int = 2048,
            fp_radius: int = 2,
            batch_size: int = 50_000,
            num_permutations: int = 64,
            band_size: int = 4,
            max_bucket_size: int = 1000,
            max_candidates_per_anchor_bucket: int = 100,
            max_negative_samples_per_anchor: int = 1024,
            similarity_threshold: float = 0.7,
            dissimilarity_threshold: float = 0.1,
            seed: int = 42
    ):

        self.all_smiles = all_smiles
        self.num_smiles = len(all_smiles)

        self.fp_size = fp_size
        self.num_words = fp_size // 64
        self.fp_radius = fp_radius

        self.num_permutations = num_permutations
        self.band_size = band_size
        self.max_bucket_size = max_bucket_size
        self.max_candidates_per_anchor_bucket = max_candidates_per_anchor_bucket
        self.similarity_threshold = similarity_threshold
        self.dissimilarity_threshold = dissimilarity_threshold
        self.max_negative_samples_per_anchor = max_negative_samples_per_anchor

        self.batch_size = batch_size

        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.fp_file, self.bit_count_file, self.signatures_file = None, None, None

        self.seed = seed

        self._build_fingerprint_cache()

    def _build_fingerprint_cache(self):
        """
        Compute and cache the fingerprints, bit counts, and MinHash signatures for all SMILES strings in the input list
        as memory-mapped files on disk. This allows for memory-efficient computation of approximate and exact
        similarities during sampling without having to hold all fingerprints in memory at once.

        Fingerprints are stored as packed uint64 arrays, bit counts are stored as uint16, and MinHash signatures are
        stored as uint32. The MinHash signatures are computed using a pre-generated MinHash table of random hash values
        for each bit index in the fingerprint. The cache files are created in the specified cache directory.
        """
        minhash_table = get_minhash_table(num_permutations=self.num_permutations, fp_size=self.fp_size, seed=self.seed)

        self.fp_file = self.cache_dir / "fingerprints.uint64.memmap"
        self.bit_count_file = self.cache_dir / "bit_counts.uint16.memmap"
        self.signatures_file = self.cache_dir / "signatures.uint32.memmap"

        self.fp_file.unlink(missing_ok=True)
        self.bit_count_file.unlink(missing_ok=True)
        self.signatures_file.unlink(missing_ok=True)

        fingerprints = np.memmap(
            self.fp_file,
            dtype=np.uint64,
            mode="w+",
            shape=(self.num_smiles, self.num_words)
        )
        bit_counts = np.memmap(
            self.bit_count_file,
            dtype=np.uint16,
            mode="w+",
            shape=(self.num_smiles,)
        )
        signatures = np.memmap(
            self.signatures_file,
            dtype=np.uint32,
            mode="w+",
            shape=(self.num_smiles, self.num_permutations)
        )
        fingerprints[:] = 0
        bit_counts[:] = 0
        signatures[:] = 0

        fp_generator = rdFingerprintGenerator.GetMorganGenerator(radius=self.fp_radius, fpSize=self.fp_size)

        row_idx = 0
        for batch_start_idx in range(0, self.num_smiles, self.batch_size):
            batch_smiles = self.all_smiles[batch_start_idx:batch_start_idx + self.batch_size]
            batch_mols = [Chem.MolFromSmiles(s) for s in batch_smiles]
            batch_fps = fp_generator.GetFingerprints(batch_mols)

            for fp in batch_fps:
                arr = np.empty(self.fp_size, dtype=np.uint8)
                DataStructs.ConvertToNumpyArray(fp, arr)
                fingerprints[row_idx, :] = np.packbits(arr, bitorder="little").view(np.uint64)
                on_bits = np.fromiter(fp.GetOnBits(), dtype=np.int32)
                bit_counts[row_idx] = on_bits.size
                signatures[row_idx, :] = get_minhash_signature(on_bits, minhash_table)
                row_idx += 1

        fingerprints.flush(), bit_counts.flush(), signatures.flush()

    def sample_positive_examples(self, num_examples: int) -> Path:
        """
        For each SMILES string in the input list, sample `num_examples` positive examples (i.e., similar SMILES strings)
        using LSH candidate generation. Stores the indices of the positive examples in a memory-mapped file on disk,
        and returns the path to the file containing the positive example indices.

        Args:
            num_examples (int): The number of positive examples to sample for each SMILES string.

        Returns:
            Path: The path to the memory-mapped file containing the indices of the positive examples for each SMILES
                  string (shape: (num_smiles, num_examples), dtype: int32).
        """
        # Load cached fingerprints, bit counts, and signatures as memory-mapped files
        fingerprints = np.memmap(
            self.fp_file,
            dtype=np.uint64,
            mode="r+",
            shape=(self.num_smiles, self.num_words)
        )
        bit_counts = np.memmap(
            self.bit_count_file,
            dtype=np.uint16,
            mode="r+",
            shape=(self.num_smiles,)
        )
        signatures = np.memmap(
            self.signatures_file,
            dtype=np.uint32,
            mode="r+",
            shape=(self.num_smiles, self.num_permutations)
        )

        # Create memory-mapped files for indices, similarities, and counts of positive examples
        pos_indices = np.memmap(
            self.cache_dir / "positive_indices.int32.memmap",
            dtype=np.int32,
            mode="w+",
            shape=(self.num_smiles, num_examples)
        )
        pos_indices[:] = -1
        positive_similarities = np.memmap(
            self.cache_dir / "positive_similarities.float32.memmap",
            dtype=np.float32,
            mode="w+",
            shape=(self.num_smiles, num_examples)
        )
        positive_similarities[:] = np.nan
        positive_counts = np.memmap(
            self.cache_dir / "positive_counts.uint16.memmap",
            dtype=np.uint16,
            mode="w+",
            shape=(self.num_smiles,)
        )
        positive_counts[:] = 0

        rng = np.random.default_rng(self.seed + 1)

        num_bands = self.num_permutations // self.band_size

        for band_idx in range(num_bands):

            keys = hash_lsh_band(signatures, band_idx=band_idx, band_size=self.band_size)

            order = np.argsort(keys, kind="stable")
            sorted_keys = keys[order]
            change = np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1
            starts, ends = np.concatenate(([0], change)), np.concatenate((change, [len(keys)]))

            for start, end in zip(starts, ends):
                bucket_size = end - start
                if bucket_size < 2 or bucket_size > self.max_bucket_size:
                    continue

                bucket_indices = order[start:end]
                rng.shuffle(bucket_indices)

                for anchor_id in bucket_indices:
                    anchor_id = int(anchor_id)
                    num_anchor_positives = positive_counts[anchor_id]
                    if num_anchor_positives >= num_examples:
                        continue

                    candidate_ids = bucket_indices[bucket_indices != anchor_id]
                    if candidate_ids.size > self.max_candidates_per_anchor_bucket:
                        candidate_ids = rng.choice(
                            candidate_ids,
                            size=self.max_candidates_per_anchor_bucket,
                            replace=False
                        )

                    similarities = batch_tanimoto_similarity(fingerprints, bit_counts, anchor_id, candidate_ids)

                    accepted_mask = similarities >= self.similarity_threshold
                    if not np.any(accepted_mask):
                        continue

                    accepted_candidate_ids = candidate_ids[accepted_mask]
                    accepted_similarities = similarities[accepted_mask]
                    rank = np.argsort(-accepted_similarities, kind="stable")
                    existing = set(int(x) for x in pos_indices[anchor_id, :num_anchor_positives] if int(x) > 0)

                    for local_idx in rank:
                        candidate_id = int(accepted_candidate_ids[local_idx])
                        if candidate_id in existing:
                            continue
                        pos_indices[anchor_id, num_anchor_positives] = candidate_id
                        positive_similarities[anchor_id, num_anchor_positives] = accepted_similarities[local_idx]
                        positive_counts[anchor_id] += 1
                        existing.add(candidate_id)
                        num_anchor_positives += 1
                        if num_anchor_positives >= num_examples:
                            break

        pos_indices.flush(), positive_similarities.flush(), positive_counts.flush()

        np.save(self.cache_dir / "positive_indices.npy", pos_indices)

        # Clear cache
        (self.cache_dir / "positive_indices.int32.memmap").unlink(missing_ok=True)
        (self.cache_dir / "positive_similarities.float32.memmap").unlink(missing_ok=True)
        (self.cache_dir / "positive_counts.uint16.memmap").unlink(missing_ok=True)

        return self.cache_dir / "positive_indices.npy"

    def sample_negative_examples(self, num_examples: int) -> Path:
        """
        For each SMILES string in the input list, sample `num_examples` negative examples (i.e., dissimilar SMILES
        strings) using random rejection sampling.

        Args:
            num_examples (int): The number of negative examples to sample for each SMILES string.

        Returns:
            Path: The path to the memory-mapped file containing the indices of the negative examples for each SMILES
                  string (shape: (num_smiles, num_examples), dtype: int32).
        """
        fingerprints = np.memmap(self.fp_file, dtype=np.uint64, mode="r+", shape=(self.num_smiles, self.num_words))
        bit_counts = np.memmap(self.bit_count_file, dtype=np.uint16, mode="r+", shape=(self.num_smiles,))

        neg_indices = np.memmap(
            self.cache_dir / "negative_indices.int32.memmap",
            dtype=np.int32,
            mode="w+",
            shape=(self.num_smiles, num_examples)
        )
        neg_indices[:] = -1
        negative_similarities = np.memmap(
            self.cache_dir / "negative_similarities.float32.memmap",
            dtype=np.float32,
            mode="w+",
            shape=(self.num_smiles, num_examples)
        )
        negative_similarities[:] = np.nan
        negative_counts = np.memmap(
            self.cache_dir / "negative_counts.uint16.memmap",
            dtype=np.uint16,
            mode="w+",
            shape=(self.num_smiles,)
        )
        negative_counts[:] = 0

        rng = np.random.default_rng(self.seed + 2)
        batch_size = min(self.max_negative_samples_per_anchor, self.num_smiles)

        for anchor_id in range(self.num_smiles):

            existing = set()
            num_tested = 0

            while int(negative_counts[anchor_id]) < num_examples:
                if num_tested >= num_examples * batch_size:
                    break

                num_missing_examples = num_examples - int(negative_counts[anchor_id])
                sample_size = max(batch_size, num_missing_examples * 10)

                candidate_ids = rng.integers(low=0, high=self.num_smiles, size=sample_size, dtype=np.int64)
                candidate_ids = candidate_ids[candidate_ids != anchor_id]

                if candidate_ids.size == 0:
                    continue

                candidate_ids = np.unique(candidate_ids)

                if existing:
                    candidate_ids = candidate_ids[~np.isin(candidate_ids, np.fromiter(existing, dtype=np.int64))]

                if candidate_ids.size == 0:
                    continue

                num_tested += candidate_ids.size

                similarities = batch_tanimoto_similarity(fingerprints, bit_counts, anchor_id, candidate_ids)

                accepted_mask = similarities <= self.dissimilarity_threshold
                if not np.any(accepted_mask):
                    continue

                accepted_candidate_ids = candidate_ids[accepted_mask]
                accepted_similarities = similarities[accepted_mask]
                rank = np.argsort(accepted_similarities, kind="stable")
                for local_idx in rank:
                    candidate_id = int(accepted_candidate_ids[local_idx])
                    if candidate_id in existing:
                        continue
                    col_idx = int(negative_counts[anchor_id])
                    neg_indices[anchor_id, col_idx] = candidate_id
                    negative_similarities[anchor_id, col_idx] = float(accepted_similarities[local_idx])
                    negative_counts[anchor_id] += 1
                    existing.add(candidate_id)
                    if int(negative_counts[anchor_id]) >= num_examples:
                        break

        neg_indices.flush(), negative_similarities.flush(), negative_counts.flush()

        np.save(self.cache_dir / "negative_indices.npy", neg_indices)

        # Clear cache
        (self.cache_dir / "negative_indices.int32.memmap").unlink(missing_ok=True)
        (self.cache_dir / "negative_similarities.float32.memmap").unlink(missing_ok=True)
        (self.cache_dir / "negative_counts.uint16.memmap").unlink(missing_ok=True)

        return self.cache_dir / "negative_indices.npy"
