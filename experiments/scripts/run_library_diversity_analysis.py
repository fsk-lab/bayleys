from pathlib import Path
import argparse
import logging
import time
import json

from tqdm import tqdm
import numpy as np
import pandas as pd
from bayleys.config import BayleysConfig
from bayleys.molecule_library import MoleculeLibraryConfig, MoleculeLibrary
from bayleys.encoders import MorganFingerprintEncoder, MoleculeEncoderConfig
from bayleys.utils.cache import load_cached_embeddings, cache_embeddings


def calculate_tanimoto_distances(fp_batch_1: np.ndarray, fp_batch_2: np.ndarray) -> np.ndarray:
    """
    Calculate the Tanimoto distance between two batches of binary fingerprints.

    Args:
        fp_batch_1 (np.ndarray): First batch of binary fingerprints (shape: [batch_size, num_bits]).
        fp_batch_2 (np.ndarray): Second batch of binary fingerprints (shape: [batch_size, num_bits]).

    Returns:
        np.ndarray: Tanimoto distances between the two batches (shape: [batch_size]).
    """
    # Pack bits into bytes for efficient computation
    fp_batch_1_packed = np.packbits(fp_batch_1.astype(np.bool_), axis=1, bitorder="little")
    fp_batch_2_packed = np.packbits(fp_batch_2.astype(np.bool_), axis=1, bitorder="little")

    # Calculate intersection and union
    intersection_bytes = np.bitwise_and(fp_batch_1_packed, fp_batch_2_packed)
    intersection = POPCOUNT_UINT8[intersection_bytes].sum(axis=1, dtype=np.uint32)

    union_bytes = np.bitwise_or(fp_batch_1_packed, fp_batch_2_packed)
    union = POPCOUNT_UINT8[union_bytes].sum(axis=1, dtype=np.uint32)

    # Calculate Tanimoto similarity
    similarities = np.divide(intersection, union, out=np.ones(fp_batch_1.shape[0], dtype=np.float32), where=union != 0)

    return 1.0 - similarities  # Return Tanimoto distances


def calculate_cosine_distances(emb_batch_1: np.ndarray, emb_batch_2: np.ndarray) -> np.ndarray:
    """
    Calculate the cosine distance between two batches of embeddings.

    Args:
        emb_batch_1 (np.ndarray): First batch of embeddings (shape: [batch_size, embedding_dim]).
        emb_batch_2 (np.ndarray): Second batch of embeddings (shape: [batch_size, embedding_dim]).

    Returns:
        np.ndarray: Cosine distances between the two batches (shape: [batch_size]).
    """
    dot_product = np.einsum('ij,ij->i', emb_batch_1, emb_batch_2)
    norms = np.linalg.norm(emb_batch_1, axis=1) * np.linalg.norm(emb_batch_2, axis=1)
    similarities = dot_product / np.maximum(norms, 1e-10)  # Avoid division by zero
    similarities = np.clip(similarities, -1.0, 1.0)  # Ensure values are in the valid range for arccos
    return 1.0 - similarities  # Return cosine distances


def sample_internal_pairs(num_elements: int, random_seed: int = 42) -> tuple[int, np.ndarray, np.ndarray]:
    """
    Sample pairs of indices from a set of elements.

    Args:
        num_elements (int): Total number of elements to sample from.
        random_seed (int): Seed for the random number generator.

    Returns:
        int: Number of sampled pairs (up to NUM_SAMPLES).
        np.ndarray: Array of shape (NUM_SAMPLES, ) containing sampled indices (first element of the pair).
        np.ndarray: Array of shape (NUM_SAMPLES, ) containing sampled indices (second element of the pair).
    """
    rng = np.random.default_rng(seed=random_seed)
    num_samples = min(NUM_SAMPLES, num_elements * (num_elements - 1) // 2)
    indices_1 = rng.integers(0, num_elements, size=num_samples)
    indices_2 = rng.integers(0, num_elements - 1, size=num_samples)
    indices_2 += (indices_2 >= indices_1)  # Ensure no self-pairs
    return num_samples, indices_1, indices_2


def sample_external_pairs(num_elements_1: int, num_elements_2: int, random_seed: int = 42) -> tuple[int, np.ndarray, np.ndarray]:
    """
    Sample pairs of indices from two different sets of elements.

    Args:
        num_elements_1 (int): Total number of elements in the first set.
        num_elements_2 (int): Total number of elements in the second set.
        random_seed (int): Seed for the random number generator.

    Returns:
        int: Number of sampled pairs (up to NUM_SAMPLES).
        np.ndarray: Array of shape (NUM_SAMPLES, ) containing sampled indices from the first set.
        np.ndarray: Array of shape (NUM_SAMPLES, ) containing sampled indices from the second set.
    """
    rng = np.random.default_rng(seed=random_seed)
    num_samples = min(NUM_SAMPLES, num_elements_1 * num_elements_2)
    flat_indices = rng.choice(num_elements_1 * num_elements_2, size=num_samples, replace=False)
    indices_1 = flat_indices // num_elements_2
    indices_2 = flat_indices % num_elements_2
    return num_samples, indices_1, indices_2


POPCOUNT_UINT8 = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)
NUM_SAMPLES = 5_000_000
BATCH_SIZE = 200_000

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Analyze the diversity of molecule libraries in LLM embedding space.")
    parser.add_argument("--library", type=str, required=True)
    args = parser.parse_args()

    with open(Path(__file__).parent / "working_directories.json", "r") as f:
        working_directories = json.load(f)

    data_dir = Path(working_directories["data"])
    embedding_dir = data_dir / "cache" / "embeddings"
    results_dir = Path(working_directories["results"])
    encoder_config_dir = Path(working_directories["encoder_configs"])
    training_config_dir = Path(working_directories["training_configs"])

    bayleys_config = BayleysConfig(
        log_level="DEBUG",
        log_file=None,
        cache_dir=data_dir / "cache",
        device="cpu"
    )
    bayleys_config.apply()
    logger = logging.getLogger("bayleys")

    library_config = MoleculeLibraryConfig.from_json(data_dir / "libraries" / f"{args.library}.json")
    library = MoleculeLibrary.from_config(library_config, local_dir=data_dir)
    library_targets = library.labels.numpy().flatten()
    library_targets = (library_targets - np.min(library_targets)) / (np.max(library_targets) - np.min(library_targets))
    if library.objective_direction == "min":
        library_targets = 1.0 - library_targets
    logger.info(f"Loaded library {library.name} with {len(library)} molecules.")

    reference_library = pd.read_csv(data_dir / "libraries" / "zinc15_leadlike_random_25k.csv")
    reference_smiles = reference_library["smiles"].to_numpy()

    reference_fps = np.load(data_dir / "cache" / "embeddings" / "zinc15_leadlike_random_25k_morgan2_2048.npy")

    encoders = ("MorganFingerprint-2-2048", "ChemBERTa-Zinc-mean", "MolFormer-XL-mean", "SmiTed", "T5Chem-mean")

    for enc_name in encoders:

        emb_data_file = embedding_dir / f"Emb_{library.name}_{enc_name}.pt"
        encoder_config_file = encoder_config_dir / f"{enc_name}.json"
        if not encoder_config_file.exists():
            raise ValueError(f"Encoder configuration file {encoder_config_file} not found.")
        encoder_config = MoleculeEncoderConfig.from_json(encoder_config_file)
        encoder = encoder_config.get_encoder()

        try:
            library_embeddings = load_cached_embeddings(encoder.name, library.name)
            logger.info(f"Loaded cached embeddings for library {library.name} with encoder {encoder.name}.")
        except FileNotFoundError:
            logger.warning(f"Embedding file {emb_data_file} does not exist. Encoding library {library.name} with encoder {enc_name}.")
            library_embeddings = encoder.encode(library.smiles)
            cache_embeddings(library_embeddings, encoder.name, library.name)
            library_embeddings = library_embeddings.cpu().numpy()

        if enc_name == "MorganFingerprint-2-2048":
            reference_embeddings = reference_fps
            logger.info(f"Using precomputed Morgan fingerprint embeddings for reference data.")
        else:
            reference_embeddings = encoder.encode(reference_smiles).numpy()
            logger.info(f"Computed embeddings for reference data with encoder {encoder.name}.")

        # Calculate pairwise distances within the reference embeddings
        num_samples, ref_indices_1, ref_indices_2 = sample_internal_pairs(reference_embeddings.shape[0], random_seed=42)
        distances_within_reference = np.empty(num_samples, dtype=np.float32)

        start_time = time.time()
        for start in tqdm(range(0, num_samples, BATCH_SIZE), desc=f"Computing pairwise distances within reference data"):
            stop = min(start + BATCH_SIZE, num_samples)
            reference_batch_1 = reference_embeddings[ref_indices_1[start:stop]]
            reference_batch_2 = reference_embeddings[ref_indices_2[start:stop]]

            if isinstance(encoder, MorganFingerprintEncoder):
                distances_within_reference[start:stop] = calculate_tanimoto_distances(reference_batch_1, reference_batch_2)
            else:
                distances_within_reference[start:stop] = calculate_cosine_distances(reference_batch_1, reference_batch_2)

        logger.info(f"Computed pairwise distances within reference data for {num_samples} pairs in {time.time() - start_time:.1f} seconds.")
        np.save(results_dir / f"DistanceWithinReference_{library.name}_{enc_name}_None.npy", distances_within_reference)

        # Calculate pairwise distances between library and reference embeddings
        num_samples, library_indices, reference_indices = sample_external_pairs(library_embeddings.shape[0], reference_embeddings.shape[0], random_seed=42)
        distances = np.empty(num_samples, dtype=np.float32)

        start_time = time.time()
        for start in tqdm(range(0, num_samples, BATCH_SIZE), desc=f"Computing pairwise distances between library {library.name} and reference data"):
            stop = min(start + BATCH_SIZE, num_samples)
            library_batch = library_embeddings[library_indices[start:stop]]
            reference_batch = reference_embeddings[reference_indices[start:stop]]

            if isinstance(encoder, MorganFingerprintEncoder):
                distances[start:stop] = calculate_tanimoto_distances(library_batch, reference_batch)
            else:
                distances[start:stop] = calculate_cosine_distances(library_batch, reference_batch)

        logger.info(f"Computed pairwise distances for {num_samples} pairs in {time.time() - start_time:.1f} seconds.")
        np.save(results_dir / f"DistanceToReference_{library.name}_{enc_name}_None.npy", distances)

        # Calculate pairwise distances within the library embeddings
        num_samples, indices_1, indices_2 = sample_internal_pairs(library_embeddings.shape[0], random_seed=42)
        distances_within_library = np.empty(num_samples, dtype=np.float32)
        functional_distances_within_library = np.empty(num_samples, dtype=np.float32)

        start_time = time.time()
        for start in tqdm(range(0, num_samples, BATCH_SIZE), desc=f"Computing pairwise distances within library {library.name}"):
            stop = min(start + BATCH_SIZE, num_samples)
            library_batch_1 = library_embeddings[indices_1[start:stop]]
            library_batch_2 = library_embeddings[indices_2[start:stop]]

            if isinstance(encoder, MorganFingerprintEncoder):
                distances_within_library[start:stop] = calculate_tanimoto_distances(library_batch_1, library_batch_2)
            else:
                distances_within_library[start:stop] = calculate_cosine_distances(library_batch_1, library_batch_2)

            functional_distances_within_library[start:stop] = np.abs(library_targets[indices_1[start:stop]] - library_targets[indices_2[start:stop]])

        logger.info(f"Computed pairwise distances within library for {num_samples} pairs in {time.time() - start_time:.1f} seconds.")
        np.save(results_dir / f"DistanceWithinLibrary_{library.name}_{encoder.name}_None.npy", distances_within_library)
        np.save(results_dir / f"FunctionalDistanceWithinLibrary_{library.name}_{encoder.name}_None.npy", functional_distances_within_library)

        # Calculate distribution of top-1% candidates
        top_1_percent_mask = library_targets >= np.percentile(library_targets, 99)
        top_1_percent_embeddings = library_embeddings[top_1_percent_mask]
        top_1_percent_targets = library_targets[top_1_percent_mask]

        num_samples, top_indices_1, top_indices_2 = sample_internal_pairs(top_1_percent_embeddings.shape[0], random_seed=42)
        top_1_percent_distances = np.empty(num_samples, dtype=np.float32)
        top_1_percent_functional_distances = np.empty(num_samples, dtype=np.float32)

        start_time = time.time()
        for start in tqdm(range(0, num_samples, BATCH_SIZE), desc=f"Computing pairwise distances within top 1% candidates of library {library.name}"):
            stop = min(start + BATCH_SIZE, num_samples)
            top_batch_1 = top_1_percent_embeddings[top_indices_1[start:stop]]
            top_batch_2 = top_1_percent_embeddings[top_indices_2[start:stop]]

            if isinstance(encoder, MorganFingerprintEncoder):
                top_1_percent_distances[start:stop] = calculate_tanimoto_distances(top_batch_1, top_batch_2)
            else:
                top_1_percent_distances[start:stop] = calculate_cosine_distances(top_batch_1, top_batch_2)

            top_1_percent_functional_distances[start:stop] = np.abs(top_1_percent_targets[top_indices_1[start:stop]] - top_1_percent_targets[top_indices_2[start:stop]])

        logger.info(f"Computed pairwise distances within top 1% candidates for {num_samples} pairs in {time.time() - start_time:.1f} seconds.")
        np.save(results_dir / f"DistanceWithinTop1Percent_{library.name}_{encoder.name}_None.npy", top_1_percent_distances)
        np.save(results_dir / f"FunctionalDistanceWithinTop1Percent_{library.name}_{encoder.name}_None.npy", top_1_percent_functional_distances)