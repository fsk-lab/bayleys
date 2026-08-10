import random
import numpy as np
import torch


def set_random_seeds(seed: int):
    """
    Sets the random seeds for reproducibility across various libraries.

    Args:
        seed: The random seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
