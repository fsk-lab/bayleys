from typing import Literal
import numpy as np

from ..molecule_library import MoleculeLibrary


def get_seed_experiment_indices(
        library: MoleculeLibrary,
        num_experiments: int,
        strategy: Literal["random"] = "random",
        base_seed: int = 42,
) -> list[int]:
    """
    For a given virtual library, selects initial experiments based on the specified seeding strategy.

    Args:
        library: MoleculeLibrary object containing the candidate molecules.
        num_experiments: Number of initial experiments to select.
        strategy: Strategy for selecting initial experiments. Currently implemented: "random"
        base_seed: Base random seed for reproducibility.

    Returns:
        List of indices corresponding to the selected initial experiments.
    """
    if strategy == "random":
        rng = np.random.default_rng(seed=base_seed)
        selected_indices = rng.choice(len(library), size=num_experiments, replace=False).tolist()
        return selected_indices
    else:
        raise NotImplementedError(f"Seeding strategy '{strategy}' is not implemented.")