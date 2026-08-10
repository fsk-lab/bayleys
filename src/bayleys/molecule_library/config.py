from typing import Literal
from dataclasses import dataclass

from ..utils import AbstractConfig


@dataclass
class MoleculeLibraryConfig(AbstractConfig):
    """
    Configuration class for molecule libraries. Can be used to load datasets from files.
    """
    name: str
    objective_direction: Literal["min", "max"]

    data_file: str  # relative to the project's local `data/` directory
    data_directory: str = None  # optional, specifies the project`s local `data/` subdirectory

    has_labels: bool = True
    label_name: str = None

    embedding_name: str = None

    smiles_column: str = "SMILES"
    max_smiles_length: int = None
    building_blocks_columns: str | list[str] = None
    label_column: str = "label"

    source: str = None
    description: str = None

    # Dataset Statistics
    size: int = None
    quantiles: dict = None
    top_k: dict = None

    def __post_init__(self):

        if self.quantiles is None:
            self.quantiles = {}
        if self.top_k is None:
            self.top_k = {}
