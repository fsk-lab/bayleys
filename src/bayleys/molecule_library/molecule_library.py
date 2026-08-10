from typing import Optional, Literal
from pathlib import Path
import copy
import pandas as pd
import numpy as np
import pyarrow as pa
import torch
from torch import Tensor
from datasets import Dataset, DatasetInfo, Features, Value, Sequence

from .config import MoleculeLibraryConfig

from ..utils import TensorLike, get_timestamp, tensor_to_pyarrow


class MoleculeLibrary(torch.utils.data.Dataset):
    """
    Object that represents a library of molecules, optionally with associated labels and encodings. Follows
    the torch Dataset API.

    Stores the following four main attributes for a library:
        0. `smiles`: np.ndarray of SMILES strings representing the molecules in the library. Shape: (N,)
        1. `building_blocks`: Optional tensor of building block information associated with each molecule in the
                              library. Shape: (N, ...).
        2. `embeddings`: Optional precomputed embeddings for the molecules in the library. Shape: (N, D)
        3. `labels`: Optional labels associated with each molecule in the library. Shape: (N, C)

    These attributes can be obtained jointly (via __getitem__) or separately (via properties).

    Args:
        smiles (Optional[list, np.ndarray, torch.Tensor]): List/array/tensor of SMILES strings representing the
                                                           molecules in the virtual library. Shape: (N,)
        building_blocks (Optional[list, np.ndarray, torch.Tensor]): List/array/tensor of building block information
                                                                    associated with each molecule in the library.
                                                                    Shape: (N, ...)
        embeddings (Optional[TensorLike]): Precomputed embeddings for the molecules in the library. Shape: (N, D)
        labels (Optional[TensorLike]): Labels associated with each molecule in the library. Shape: (N,) or (N, C)
    """

    def __init__(
            self,
            dataset_config: MoleculeLibraryConfig,
            smiles: Optional[list | np.ndarray | Tensor] = None,
            building_blocks: Optional[list | np.ndarray | Tensor] = None,
            embeddings: Optional[TensorLike] = None,
            labels: Optional[TensorLike] = None,
    ):
        """
        Instantiates a MoleculeLibrary from the provided data. Validates the shape and types of the inputs, and converts
        the data to the specified formats (np.ndarray for the SMILES strings, torch.Tensor for the rest).
        """
        self.dataset_config = dataset_config

        # Validate and store smiles
        if smiles is not None:
            smiles = np.array(smiles).flatten()
            self.dataset_config.size = len(smiles)
        self._smiles = smiles

        # Validate and store building_blocks, if provided. Otherwise store None.
        if building_blocks is not None:
            if not isinstance(building_blocks, Tensor):
                building_blocks = torch.tensor(building_blocks)
            if building_blocks.shape[0] != self._smiles.shape[0]:
                raise ValueError("Length of building_blocks must match length of smiles")
        self._building_blocks = building_blocks

        # Validate and store embeddings, if provided. Otherwise store None.
        if embeddings is not None:
            if not isinstance(embeddings, Tensor):
                embeddings = torch.tensor(embeddings)
            if embeddings.shape[0] != self._smiles.shape[0]:
                raise ValueError("Length of embeddings must match length of smiles")
        self._embeddings = embeddings

        # Validate and store labels, if provided, also validate objective_direction. Otherwise store None.
        if labels is not None:
            if not isinstance(labels, Tensor):
                labels = torch.tensor(labels)
            if labels.shape[0] != self._smiles.shape[0]:
                raise ValueError("Length of labels must match length of smiles")
        self._labels = labels

        self.device = torch.device("cpu")

    @classmethod
    def from_config(
            cls,
            config: MoleculeLibraryConfig,
            local_dir: Optional[Path] = None,
            read_only: bool = False
    ) -> "MoleculeLibrary":
        """
        Instantiates a VirtualLibrary from a local archive directory. The configuration object specifies the dataset
        and all associated metadata.

        Args:
            config: MoleculeLibraryConfig object specifying the dataset to load and associated metadata.
            local_dir: Path to the local directory containing the dataset archive. If None, requires a data directory
                       to be specified in the config object.
            read_only: Whether to open the dataset in read-only mode. Default is False. In read-only mode, no actual
                       data is loaded, only the metadata from the config object.

        Returns:
            MoleculeLibrary: Instantiated VirtualLibrary object
        """
        if read_only:
            return cls(dataset_config=config)

        if local_dir is None:
            if config.data_directory is None:
                raise ValueError("Either local_dir must be provided, or data_directory must be specified in the "
                                 "config object.")
            local_dir = Path(config.data_directory)
        else:
            config.data_directory = str(local_dir)

        dataset_file = local_dir / config.data_file

        if not dataset_file.exists() or not dataset_file.is_file():
            raise FileNotFoundError(f"Dataset file {dataset_file} not found.")
        suffix = "".join(dataset_file.suffixes)
        if suffix not in (".csv", ".csv.gz"):
            raise ValueError(f"Unsupported  file format: {dataset_file.suffix}. Only .csv and .csv.gz are supported.")

        df = pd.read_csv(dataset_file)

        smiles = df[config.smiles_column].to_numpy().astype(str)

        if config.building_blocks_columns is not None:
            building_blocks = torch.tensor(df[config.building_blocks_columns].to_numpy()).type(torch.get_default_dtype())
        else:
            building_blocks = None

        if config.embedding_name is not None:
            config.embedding_name = None  # embeddings are not loaded from file, only computed and cached separately

        if config.label_column is not None:
            labels = torch.tensor(df[config.label_column].to_numpy()).type(torch.get_default_dtype()).unsqueeze(-1)
        else:
            labels, objective_direction = None, None

        return cls(
            dataset_config=config,
            smiles=smiles,
            building_blocks=building_blocks,
            labels=labels,
        )

    @classmethod
    def from_dict(cls, data: dict) -> "MoleculeLibrary":
        """
        Instantiates a MoleculeLibrary from a dictionary representation, as generated by the `to_dict(include=True)`
        method.

        Args:
            data (dict): Dictionary representation of the MoleculeLibrary.

        Returns:
            MoleculeLibrary: Instantiated MoleculeLibrary object.
        """
        dataset_config = MoleculeLibraryConfig.from_dict(data["config"])

        smiles = data.get("smiles")

        building_blocks = None
        if "building_blocks" in data and data["building_blocks"] is not None:
            building_blocks = torch.tensor(data["building_blocks"])

        embeddings = None
        if "embeddings" in data and data["embeddings"] is not None:
            embeddings = torch.tensor(data["embeddings"])

        labels = None
        if "labels" in data and data["labels"] is not None:
            labels = torch.tensor(data["labels"])

        return cls(
            dataset_config=dataset_config,
            smiles=smiles,
            building_blocks=building_blocks,
            embeddings=embeddings,
            labels=labels,
        )

    def to_dict(self, include_data: bool = False) -> dict:
        """
        Converts the MoleculeLibrary to a dictionary representation.

        Args:
            include_data (bool): Whether to include the actual data (smiles and labels) in the dictionary. Default is
            False.

        Returns:
            dict: Dictionary representation of the MoleculeLibrary.
        """
        data = {"config": self.dataset_config.to_dict()}

        if include_data:
            data["smiles"] = self._smiles.tolist() if self._smiles is not None else None
            # data["building_blocks"] = self._building_blocks.tolist() if self._building_blocks is not None else None
            # data["embeddings"] = self._embeddings.tolist() if self._embeddings is not None else None
            data["labels"] = self._labels.tolist() if self._labels is not None else None

        return data

    def to_huggingface_dataset(
            self,
            include_building_blocks: bool = False,
            include_embeddings: bool = False,
            include_labels: bool = False
    ) -> Dataset:
        """
        Converts the MoleculeLibrary to a HuggingFace Dataset object.

        Args:
            include_building_blocks (bool): Whether to include building block information in the dataset.
            include_embeddings (bool): Whether to include embeddings in the dataset.
            include_labels (bool): Whether to include labels in the dataset.

        Returns:
            Dataset: HuggingFace Dataset representation of the MoleculeLibrary.
        """
        if self._smiles is None:
            raise ValueError("Cannot convert to HuggingFace Dataset: SMILES data is not available.")

        features = {"smiles": Value("string")}
        columns = {"smiles": pa.array(self._smiles.tolist(), type=pa.string())}

        if include_building_blocks and self._building_blocks is not None:
            features["building_blocks"] = Sequence(Value("float32"), length=self._building_blocks.shape[1])
            columns["building_blocks"] = tensor_to_pyarrow(self._building_blocks)

        if include_embeddings and self._embeddings is not None:
            features["embeddings"] = Sequence(Value("float32"), length=self._embeddings.shape[1])
            columns["embeddings"] = tensor_to_pyarrow(self._embeddings)

        if include_labels and self._labels is not None:
            features["labels"] = Sequence(Value("float32"), length=self._labels.shape[1])
            columns["labels"] = tensor_to_pyarrow(self._labels)

        table = pa.table(columns)
        table_info = DatasetInfo(features=Features(features))
        dataset = Dataset(table, info=table_info)
        dataset.with_format("torch")

        return dataset

    def train_test_split(
            self,
            test_size: float = 0.1,
            random_seed: Optional[int] = None
    ) -> tuple["MoleculeLibrary", "MoleculeLibrary"]:
        """
        Splits the MoleculeLibrary into a training and test set.

        Args:
            test_size (float): Proportion of the dataset to include in the test split. Default is 0.1.
            random_seed (Optional[int]): Random seed for reproducibility. If None, the split will be random each time.

        Returns:
            tuple[MoleculeLibrary, MoleculeLibrary]: A tuple containing the training and test splits of the dataset.
        """
        if self._smiles is None:
            raise ValueError("Cannot split dataset: SMILES data is not available.")

        rng = np.random.default_rng(seed=random_seed)
        num_samples = len(self)
        indices = rng.permutation(num_samples)
        split_idx = int(num_samples * (1 - test_size))

        train_indices, test_indices = indices[:split_idx], indices[split_idx:]

        train_smiles, train_building_blocks, train_embeddings, train_labels = self[train_indices]
        train_library = MoleculeLibrary(
            dataset_config=copy.deepcopy(self.dataset_config),
            smiles=train_smiles,
            building_blocks=train_building_blocks,
            embeddings=train_embeddings,
            labels=train_labels
        )
        test_smiles, test_building_blocks, test_embeddings, test_labels = self[test_indices]
        test_library = MoleculeLibrary(
            dataset_config=copy.deepcopy(self.dataset_config),
            smiles=test_smiles,
            building_blocks=test_building_blocks,
            embeddings=test_embeddings,
            labels=test_labels
        )

        return train_library, test_library

    def __getitem__(self, item) -> tuple[Optional[np.ndarray], Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        """
        Implements the getitem method of a torch Dataset. Returns the SMILES string, building blocks, embeddings, and
        labels for the indexed item(s).

        Args:
            item (int, slice, list, torch.Tensor): Index of the item(s) to retrieve.

        Returns:
            Optional[np.ndarray]: SMILES string(s) of the indexed item(s). Shape: (,) or (M,)
            Optional[Tensor]: Building block information of the indexed item(s). Shape: (...) or (M, ...) or None
            Optional[Tensor]: Embeddings of the indexed item(s). Shape: (D,) or (M, D) or None
            Optional[Tensor]: Labels of the indexed item(s). Shape: (C,) or (M, C) or None
        """
        smiles = self._smiles[item] if self._smiles is not None else None
        building_blocks = self._building_blocks[item, ...] if self._building_blocks is not None else None
        embeddings = self._embeddings[item, ...] if self._embeddings is not None else None
        labels = self._labels[item, ...] if self._labels is not None else None

        return smiles, building_blocks, embeddings, labels

    @classmethod
    def new(
            cls,
            dataset_name: str,
            objective_name: str,
            objective_direction: Literal["min", "max"],
            smiles_dtype: type = str
    ) -> "MoleculeLibrary":
        """
        Creates an empty MoleculeLibrary object (e.g. for running a new optimization campaign).

        Returns:
            MoleculeLibrary: Empty MoleculeLibrary object
        """
        config = MoleculeLibraryConfig(
            name=dataset_name,
            data_file=f"{dataset_name}.csv.gz",
            objective_direction=objective_direction,
            label_name=objective_name,
            smiles_column="SMILES",
            label_column=objective_name,
            source=f"generated from scratch on {get_timestamp()}"
        )

        return MoleculeLibrary(
            dataset_config=config,
            smiles=np.array([], dtype=smiles_dtype),
            labels=torch.zeros((0, 1)),
        )

    def __len__(self):

        if self.dataset_config.size is None:
            raise RuntimeError("Dataset size is not defined. The dataset may be in read-only mode without data.")
        return self.dataset_config.size

    def pop(self, item):
        """
        Implements the pop method of a list for the SMILESDataset. Removes the indexed items from the dataset and
        returns them.

        Args:
            item: Index of the item(s) to pop
        """
        if isinstance(item, int):
            item = [item]
        values = self[item]
        mask = torch.ones(len(self), dtype=torch.bool)
        mask[item] = False

        if self._smiles is not None:
            self._smiles = self._smiles[mask.numpy()]
        if self._building_blocks is not None:
            self._building_blocks = self._building_blocks[mask, ...]
        if self._embeddings is not None:
            self._embeddings = self._embeddings[mask, ...]
        if self._labels is not None:
            self._labels = self._labels[mask, ...]

        self._update_dataset_statistics()

        return values

    def extend(
            self,
            smiles: np.ndarray,
            building_blocks: Optional[Tensor] = None,
            embeddings: Optional[Tensor] = None,
            labels: Optional[Tensor] = None
        ) -> None:
        """
        Appends a new set of `M` molecules to the dataset.

        Args:
            smiles: np.ndarray of SMILES strings to add to the dataset. Shape: (M,)
            building_blocks: Optional tensor of building block information associated with each molecule to add.
                             Shape: (M, ...)
            embeddings: Optional tensor of precomputed embeddings for the molecules to add. Shape: (M, D)
            labels: Optional tensor of labels associated with each molecule to add. Shape: (M,) or (M, C)
        """
        if self._smiles is None:
            raise AttributeError("Cannot extend dataset: original dataset has no SMILES data (likely read-only mode).")

        # Add smiles
        smiles = np.array(smiles).flatten()
        self._smiles = np.concatenate([self._smiles, smiles], axis=0)
        self.dataset_config.size = len(self._smiles)

        # Validate and add building_blocks, if provided
        if building_blocks is not None:
            if not isinstance(building_blocks, Tensor):
                building_blocks = torch.tensor(building_blocks)
            if building_blocks.shape[0] != smiles.shape[0]:
                raise ValueError("Length of building_blocks must match length of smiles")
            if self._building_blocks is None:
                if len(self) == len(smiles):  # original dataset was empty
                    self._building_blocks = building_blocks
                else:
                    raise AttributeError("Can't extend building_blocks if the original dataset has no building_blocks.")
            elif self._building_blocks.shape[1:] != building_blocks.shape[1:]:
                raise ValueError("Shape of building_blocks must match shape of existing building_blocks.")
            else:
                self._building_blocks = torch.cat([self._building_blocks, building_blocks], dim=0)
        else:
            if self._building_blocks is not None:
                raise ValueError("Must provide building_blocks when the original dataset has building_blocks.")

        # Validate and add embeddings, if provided
        if embeddings is not None:
            if not isinstance(embeddings, Tensor):
                embeddings = torch.tensor(embeddings)
            if embeddings.shape[0] != smiles.shape[0]:
                raise ValueError("Length of embeddings must match length of smiles")
            if self._embeddings is None:
                if len(self) == len(smiles):  # original dataset was empty
                    self._embeddings = embeddings
                else:
                    raise AttributeError("Cannot extend embeddings when the original dataset has no embeddings.")
            elif self._embeddings.shape[1:] != embeddings.shape[1:]:
                raise ValueError("Shape of embeddings must match shape of existing embeddings.")
            else:
                self._embeddings = torch.cat([self._embeddings, embeddings], dim=0)
        else:
            if self._embeddings is not None:
                raise ValueError("Must provide embeddings when the original dataset has embeddings.")

        # Validate and add labels, if provided
        if labels is not None:
            if not isinstance(labels, Tensor):
                labels = torch.tensor(labels)
            if labels.shape[0] != smiles.shape[0]:
                raise ValueError("Length of labels must match length of smiles")
            if self._labels is None:
                if len(self) == len(smiles):  # original dataset was empty
                    self._labels = labels
                else:
                    raise AttributeError("Cannot extend labels when the original dataset has no labels.")
            elif self._labels.shape[1:] != labels.shape[1:]:
                raise ValueError("Shape of labels must match shape of existing labels.")
            else:
                self._labels = torch.cat([self._labels, labels], dim=0)
        else:
            if self._labels is not None:
                raise ValueError("Must provide labels when the original dataset has labels.")

        self._update_dataset_statistics()

    def _update_dataset_statistics(self) -> None:
        """
        Updates the dataset statistics (size, quantiles, top_k) in the dataset configuration after adding / removing
        data (i.e. via the `pop` or `extend` methods).
        """
        if self._smiles is not None:
            self.dataset_config.size = len(self._smiles)
        else:
            self.dataset_config.size = None

        self.dataset_config.quantiles = {}
        self.dataset_config.top_k = {}

    def shuffle(self) -> None:
        """
        Shuffles the dataset in-place.
        """
        perm = torch.randperm(len(self))
        if self._smiles is not None:
            self._smiles = self._smiles[perm.numpy()]
        if self._building_blocks is not None:
            self._building_blocks = self._building_blocks[perm, ...]
        if self._embeddings is not None:
            self._embeddings = self._embeddings[perm, ...]
        if self._labels is not None:
            self._labels = self._labels[perm, ...]

    @property
    def smiles(self) -> np.ndarray:
        return self._smiles

    @property
    def max_smiles_length(self) -> int:

        if self.dataset_config.max_smiles_length is None:
            self.dataset_config.max_smiles_length = np.char.str_len(self._smiles.flatten()).max().item()

        return self.dataset_config.max_smiles_length

    @property
    def embeddings(self) -> Tensor:
        if self._embeddings is None:
            raise AttributeError("Encodings are not available for this dataset.")
        return self._embeddings

    @embeddings.setter
    def embeddings(self, embeddings: Tensor):
        if embeddings.shape[0] != len(self):
            raise ValueError("Embeddings and smiles must have the same length")
        self._embeddings = embeddings

    @property
    def embedding_name(self) -> Optional[str]:
        return self.dataset_config.embedding_name

    @embedding_name.setter
    def embedding_name(self, name: str):
        self.dataset_config.embedding_name = name

    @property
    def labels(self) -> Tensor:
        return self._labels

    @property
    def objective_direction(self) -> str:
        return self.dataset_config.objective_direction

    def get_quantile(self, q: float) -> float:
        """
        Returns the q-th quantile of the labels in the dataset.

        Args:
            q: Quantile to compute, between 0 and 1.

        Returns:
            float: The q-th quantile of the labels.
        """
        if self.dataset_config.has_labels is False:
            raise AttributeError("Labels are not available for this dataset.")
        if q < 0.0 or q > 1.0:
            raise ValueError("q must be between 0 and 1.")
        if self.objective_direction == "min":
            q = 1.0 - q

        if f"{q:.3f}" not in self.dataset_config.quantiles:
            if self._smiles is None:
                raise RuntimeError("Cannot compute quantiles: Dataset is in read-only mode without data.")
            self.dataset_config.quantiles[f"{q:.3f}"] = float(torch.quantile(self._labels.flatten(), q).item())

        return self.dataset_config.quantiles[f"{q:.3f}"]

    def get_top_k_threshold(self, k: int) -> float:
        """
        Returns the threshold value for the top-k labels in the dataset.

        Args:
            k: Number of top labels to consider.

        Returns:
            float: The threshold value for the top-k labels.
        """
        if self.dataset_config.has_labels is False:
            raise AttributeError("Labels are not available for this dataset.")
        if k <= 0 or k > len(self):
            raise ValueError("k must be between 1 and the number of samples in the dataset.")

        if str(k) not in self.dataset_config.top_k:
            if self._smiles is None:
                raise RuntimeError("Cannot compute top-k thresholds: Dataset is in read-only mode without data.")
            sorted_labels, _ = torch.sort(self._labels.flatten(), descending=self.objective_direction == "max")
            self.dataset_config.top_k[str(k)] = float(sorted_labels[k - 1].item())

        return self.dataset_config.top_k[str(k)]

    @property
    def factor(self) -> float:
        """
        Returns +1.0 for maximization objectives and -1.0 for minimization objectives. Can be multiplied with labels to
        always convert to a maximization objective.
        """
        if self.dataset_config.objective_direction == "min":
            return -1.0
        else:
            return 1.0

    @property
    def name(self) -> str:
        return self.dataset_config.name

    @property
    def file(self) -> str | Path:
        return self.dataset_config.data_file

    def to(self, device: torch.device) -> "MoleculeLibrary":
        """
        Moves the dataset to the specified device. Runs the .to() method on all tensor attributes.

        Args:
            device: torch.device to move the dataset to.

        Returns:
            MoleculeLibrary: self, after moving to the specified device.
        """
        self.device = device
        if self._building_blocks is not None:
            self._building_blocks = self._building_blocks.to(device)
        if self._embeddings is not None:
            self._embeddings = self._embeddings.to(device)
        if self._labels is not None:
            self._labels = self._labels.to(device)
        return self

