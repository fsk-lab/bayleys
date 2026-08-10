from typing import Optional
from pathlib import Path
import torch

from ..config import get_config


def load_cached_embeddings(encoder_name: str, library_name: str) -> Optional[torch.Tensor]:
    """
    Loads cached embeddings for a given encoder and molecule library if available.

    Args:
        encoder_name (str): Name of the encoder.
        library_name (str): Name of the molecule library.

    Returns:
        Optional[torch.Tensor]: Cached embeddings tensor if available, else None.
    """
    cache_dir = Path(get_config().cache_dir) / "embeddings"
    cache_file = cache_dir / f"Emb_{library_name}_{encoder_name}.pt"

    if cache_file.exists():
        return torch.load(cache_file)
    raise FileNotFoundError(f"No cached embeddings found for encoder '{encoder_name}' and library '{library_name}'.")


def cache_embeddings(embeddings: torch.Tensor, encoder_name: str, library_name: str):
    """
    Caches embeddings for a given encoder and molecule library.

    Args:
        embeddings (torch.Tensor): Embeddings tensor to cache.
        encoder_name (str): Name of the encoder.
        library_name (str): Name of the molecule library.
    """
    cache_dir = Path(get_config().cache_dir) / "embeddings"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"Emb_{library_name}_{encoder_name}.pt"

    torch.save(embeddings, cache_file)
