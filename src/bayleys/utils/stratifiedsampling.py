from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def stratified_sampling(
        file_path: Path,
        score_col: str = "score",
        sample_frac: float = 0.02,
        n_bins: int = 500,
        chunksize: int = 1_000_000,
        random_state: int = 42,
):
    """
    Stratified sampling on a CSV or CSV.GZ file in streaming mode (chunked).

    Args:
        file_path: path to CSV or CSV.GZ
        score_col: column used for stratification
        sample_frac: fraction of rows to sample per bin
        n_bins: number of bins
        chunksize: number of rows per chunk
        random_state: RNG seed
    Returns:
        concatenated sampled DataFrame
    """
    rng = np.random.default_rng(random_state)
    sampled_chunks = []

    # 1️⃣ erste Pass: min/max Berechnung für Bin-Edges
    min_score = float('inf')
    max_score = float('-inf')
    for chunk in pd.read_csv(file_path, usecols=[score_col], chunksize=chunksize, engine="c", compression="infer"):
        min_score = min(min_score, chunk[score_col].min())
        max_score = max(max_score, chunk[score_col].max())

    bin_edges = np.linspace(min_score, max_score, n_bins + 1)

    # 2️⃣ Hauptloop: Sampling pro Chunk
    for chunk in pd.read_csv(file_path, chunksize=chunksize, engine="c", compression="infer"):
        scores = chunk[score_col].to_numpy()
        valid_mask = ~np.isnan(scores)
        if not np.any(valid_mask):
            continue

        bin_ids = np.searchsorted(bin_edges, scores[valid_mask], side="right") - 1
        bin_ids[bin_ids == n_bins] = n_bins - 1

        perm = rng.permutation(len(bin_ids))
        bin_ids = bin_ids[perm]
        valid_indices = np.nonzero(valid_mask)[0][perm]

        order = np.argsort(bin_ids, kind="stable")
        bin_ids = bin_ids[order]
        valid_indices = valid_indices[order]

        _, bin_counts = np.unique(bin_ids, return_counts=True)
        n_samples_per_bin = (bin_counts * sample_frac).astype(int)

        mask = np.zeros_like(bin_ids, dtype=bool)
        start = 0
        for bin_size, n_samples in zip(bin_counts, n_samples_per_bin):
            if n_samples > 0:
                mask[start:start + n_samples] = True
            start += bin_size

        sampled_chunks.append(chunk.iloc[valid_indices[mask]])

    if sampled_chunks:
        return pd.concat(sampled_chunks, ignore_index=True)
    else:
        return pd.DataFrame(columns=pd.read_csv(file_path, nrows=0, engine="c").columns)


def weighted_sampling(
        raw_dataset,
        score_col: str = 'score',
        sample_frac: float = 0.2,
        n_bins: int = 500,
        bottom_bin_frac: float = 0.1,
        cut_frac: float = 0.25
):
    """
    Stratified sampling with an option to cut out a fraction of the bottom bins.

    Args:
        raw_dataset: pandas DataFrame
        score_col: column name with continuous values
        sample_frac: fraction to sample from each bin
        n_bins: number of bins to split the score into
        bottom_bin_frac: fraction of bottom bins to consider for cutting
        cut_frac: fraction of data to remove from the bottom bins

    Returns:
        Sampled DataFrame
    """
    df_copy = raw_dataset.copy()
    df_copy['score_bin'] = pd.cut(df_copy[score_col], bins=n_bins)

    # Identify bottom bins to cut
    bottom_bins = set()
    if bottom_bin_frac > 0:
        bin_edges = df_copy['score_bin'].cat.categories
        n_bottom_bins = max(int(len(bin_edges) * bottom_bin_frac), 1)
        bottom_bins = set(bin_edges[:n_bottom_bins])

    sample = []

    for bin_cat, group in df_copy.groupby('score_bin'):
        # Remove fraction of bottom bins
        if bin_cat in bottom_bins and cut_frac > 0:
            n_to_remove = int(len(group) * cut_frac)
            if n_to_remove >= len(group):
                continue
            group = group.sample(n=len(group) - n_to_remove, random_state=42)

        # Sample fraction from the remaining group
        if sample_frac >= 1:
            stratum_sample = group.copy()
        else:
            stratum_sample = group.sample(frac=sample_frac, replace=False, random_state=42)

        sample.append(stratum_sample)

    return pd.concat(sample).drop(columns='score_bin')


def compare_sampling(
        raw_dataset,
        sampled_dataset,
        score_col: str = "score",
        n_bins: int = 50,
        raw_label: str ='Original',
        sampled_label: str = 'Sampled'
):
    """
    Plots two histograms on top of each other for comparison.

    Args:
        raw_dataset: raw DataFrame
        sampled_dataset: sampled DataFrame
        score_col: Column name to plot
        n_bins: Number of bins
        raw_label: Label for raw histogram (background)
        sampled_label: Label for sampled histogram (foreground)
    """
    plt.figure(figsize=(10, 6))

    # Background histogram
    plt.hist(raw_dataset[score_col], bins=n_bins, alpha=0.5, label=raw_label, color='blue', edgecolor='black')

    # Foreground histogram
    plt.hist(sampled_dataset[score_col], bins=n_bins, alpha=0.7, label=sampled_label, color='orange', edgecolor='black')

    plt.xlabel(score_col)
    plt.ylabel('Count')
    plt.title(f'Distribution of {score_col}')
    plt.legend()
    plt.show()