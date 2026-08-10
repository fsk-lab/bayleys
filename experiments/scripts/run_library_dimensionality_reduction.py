from pathlib import Path
import argparse
import logging
import time
import json
from tqdm import tqdm
import numpy as np
from openTSNE.sklearn import TSNE
from umap import UMAP
from bayleys.config import BayleysConfig
from bayleys.molecule_library import MoleculeLibraryConfig, MoleculeLibrary
from bayleys.encoders import MorganFingerprintEncoder
from bayleys.utils.cache import load_cached_embeddings, cache_embeddings



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Analyze the diversity of molecule libraries using UMAP and t-SNE.")
    parser.add_argument("--library", type=str, required=True)
    args = parser.parse_args()

    with open(Path(__file__).parent / "working_directories.json", "r") as f:
        working_directories = json.load(f)

    data_dir = Path(working_directories["data"])
    results_dir = Path(working_directories["results"])

    bayleys_config = BayleysConfig(
        log_level="DEBUG",
        log_file=None,
        cache_dir=data_dir / "cache",
        device="cpu"
    )
    bayleys_config.apply()
    logger = logging.getLogger("bayleys")

    reference_fps = np.load(data_dir / "cache" / "embeddings" / "zinc15_leadlike_random_25k_morgan2_2048.npy")

    umap_model = UMAP(
        n_components=2,
        n_neighbors=30,
        min_dist=0.1,
        metric="jaccard",
    )
    start_time = time.time()
    reference_umap_coords = umap_model.fit_transform(reference_fps)
    logger.info(f"Fitted a UMAP model to the reference data in {time.time() - start_time:.1f} seconds.")
    np.save(results_dir / "UMAP_Coords_ReferenceLibrary.npy", reference_umap_coords)

    # Load the library and compute its fingerprints
    library_config = MoleculeLibraryConfig.from_json(data_dir / "libraries" / f"{args.library}.json")
    library = MoleculeLibrary.from_config(library_config, local_dir=data_dir)
    logger.info(f"Loaded library {library.name} with {len(library)} molecules.")

    try:
        library_fps = load_cached_embeddings("MorganFingerprint-2-2048", library.name)
        logger.info(f"  Loaded cached Morgan fingerprints for library {library.name}.")
    except FileNotFoundError:
        logger.info(f"  Morgan fingerprints for library {library.name} not found in cache. Computing fingerprints...")
        start_time = time.time()
        fp_encoder = MorganFingerprintEncoder(radius=2, n_bits=2048)
        library_fps = fp_encoder.encode(library.smiles)
        logger.info(f"  Computed Morgan fingerprints for library {library.name} in {time.time() - start_time:.1f} seconds.")
        cache_embeddings(library_fps, "MorganFingerprint-2-2048", library.name)
    library_fps = library_fps.numpy()

    # Calculate the UMAP coordinates for the library using the fitted models and save them to disk
    start_time = time.time()
    library_umap_coords = []
    for batch_start_idx in tqdm(range(0, len(library_fps), 50_000), desc=f"Transforming library {library.name} into UMAP coordinates"):
        batch_end_idx = min(batch_start_idx + 50_000, len(library_fps))
        batch_fps = library_fps[batch_start_idx:batch_end_idx]
        batch_umap_coords = umap_model.transform(batch_fps)
        library_umap_coords.append(batch_umap_coords)
    library_umap_coords = np.vstack(library_umap_coords)
    logger.info(f" Transformed library {library.name} into UMAP coordinates in {time.time() - start_time:.1f} seconds.")
    np.save(results_dir / f"UMAP_Coords_{library.name}.npy", library_umap_coords)

    # Fit a t-SNE model to the reference data and save the coordinates to disk
    tsne_model = TSNE(
        n_components=2,
        perplexity=30,
        metric="jaccard",
        initialization="pca",
        neighbors="pynndescent",
        negative_gradient_method="fft",
        learning_rate="auto",
        n_iter=1000,
        n_jobs=-1,
        random_state=42,
        verbose=False
    )
    start_time = time.time()
    reference_tsne_coords = tsne_model.fit_transform(reference_fps)
    logger.info(f"Fitted a t-SNE model to the reference data in {time.time() - start_time:.1f} seconds.")

    start_time = time.time()
    library_tsne_coords = []
    for batch_start_idx in tqdm(range(0, len(library_fps), 50_000), desc=f"Transforming library {library.name} into t-SNE coordinates"):
        batch_end_idx = min(batch_start_idx + 50_000, len(library_fps))
        batch_fps = library_fps[batch_start_idx:batch_end_idx]
        batch_tsne_coords = tsne_model.transform(batch_fps)
        library_tsne_coords.append(batch_tsne_coords)
    library_tsne_coords = np.vstack(library_tsne_coords)
    logger.info(f" Transformed library {library.name} into t-SNE coordinates in {time.time() - start_time:.1f} seconds.")
    np.save(results_dir / f"tSNE_Coords_Reference_{library.name}.npy", np.asarray(tsne_model.embedding_).copy())
    np.save(results_dir / f"tSNE_Coords_{library.name}.npy", library_tsne_coords)



