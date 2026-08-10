from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class MemoryMappedDatasetColumn(ABC):
    """
    Abstract base class for a column of data in a memory-mapped dataset.

    Args:
        column_name (str): Name of the column.
        dtype (type | torch.dtype): Target data type for the returned data (str or torch dtype).
        file (Path): Path to the .npy file containing the data for this column.
    """

    def __init__(
            self,
            column_name: str,
            dtype: type | torch.dtype,
            file: Path,
            **kwargs
    ):
        self.name = column_name
        self.dtype = dtype
        self.data = np.load(file, mmap_mode="r", allow_pickle=True)

    @abstractmethod
    def __getitem__(self, idx) -> list | torch.Tensor:
        """
        Returns the data for the given index as a torch tensor of the specified dtype.

        Args:
            idx: Index of the example to retrieve.

        Returns:
            list | torch.Tensor: The data for the given index, returned as a list if the dtype is str, or as a torch
                                 tensor of the specified dtype otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        """
        Returns the number of examples in this column.

        Returns:
            int: The number of examples in this column.
        """
        raise NotImplementedError


class MemoryMappedVariableSizeColumn(MemoryMappedDatasetColumn):
    """
    Dataset column in which the data consists of variable-length sequences, stored in a flattened format with separate
    offset and length arrays to indicate the start and length of each sequence.

    Args:
        column_name (str): Name of the column.
        dtype (torch.dtype): Target data type (torch) for the returned tensors.
        data_file (Path): Path to the .npy file containing the flattened data for all sequences.
        offsets_file (Path): Path to the .npy file containing the offsets for each sequence.
        lengths_file (Path): Path to the .npy file containing the lengths for each sequence
    """
    def __init__(
            self,
            column_name: str,
            dtype: str | torch.dtype,
            data_file: Path,
            offsets_file: Path,
            lengths_file: Path,
    ):
        super().__init__(column_name, dtype, data_file)
        self.offsets = np.load(offsets_file, mmap_mode="r")
        self.lengths = np.load(lengths_file, mmap_mode="r")

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, idx) -> torch.Tensor:
        start, end = int(self.offsets[idx]), int(self.offsets[idx] + self.lengths[idx])
        return torch.tensor(self.data[start:end], dtype=self.dtype)


class MemoryMappedFixedSizeColumn(MemoryMappedDatasetColumn):
    """
    Dataset column in which the data is a (n_examples, dim) array of fixed-size vectors.
    """
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):

        if self.dtype == str:
            return self.data[idx].item()

        return torch.tensor(self.data[idx], dtype=self.dtype)

    @property
    def dim(self):
        return self.data.shape[1]

    @classmethod
    def from_array(
            cls,
            column_name: str,
            dtype: type | torch.dtype,
            array: list | np.ndarray | torch.Tensor,
            data_dir: Path
    ) -> "MemoryMappedFixedSizeColumn":
        """
        Creates a MemoryMappedFixedSizeColumn from a numpy array by saving the array to .npy format, which can be
        loaded in a memory-mapped way by the column class.

        Args:
            column_name (str): Name of the column.
            dtype (type | torch.dtype): Target data type for the returned data (str or torch dtype).
            array (np.ndarray): The data to be stored in the column as a numpy array.
            data_dir (Path): Directory to save the memory-mapped file to.

        Returns:
            MemoryMappedFixedSizeColumn: A MemoryMappedFixedSizeColumn instance with the data from the input array.
        """
        data_dir.mkdir(exist_ok=True, parents=True)
        data_file = data_dir / f"{column_name}.npy"
        np.save(data_file, np.asarray(array))
        return cls(column_name, dtype, data_file)


class MemoryMappedDataset(Dataset):
    """
    Dataset class for efficiently storing and loading larger pieces of data using memory-mapped files.
    """
    loss_type = "default"  # or "contrastive"

    def __init__(self, *columns: MemoryMappedDatasetColumn):

        self.column_names = []
        self.size = None
        for column in columns:
            self.add_column(column)
        self.rng = np.random.default_rng(seed=42)

    def __len__(self):

        if self.size is None:
            raise ValueError("Dataset size is not defined. At least one column must be added to the dataset.")
        return self.size

    def __getitem__(self, idx):
        """
        Returns a dictionary containing the data for the given index from all columns in the dataset.
        Currently only supports indexing by integer index, not slicing or batching.

        For contrastive learning (i.e. if self.loss_type == "contrastive"), for each column, also returns one positive
        and one negative example. For this, the dataset should have two additional columns for the indices of the
        positive and negative examples (named "_positive_indices" and "_negative_indices"), as well as an additional
        column per dataset column for the fallback example to use if no positive or negative example is available (named
        "_{column_name}_positive_fallback" and "_{column_name}_negative_fallback").

        Args:
            idx: Index of the example to retrieve.

        Returns:
            dict: A dictionary where the keys are the column names and the values are the data for the given index from
                  each column, returned as torch tensors of the specified dtype.
        """
        return_data = {}

        example_idx = None

        for column_name in self.column_names:
            if column_name.startswith("_"):
                continue
            column_data = getattr(self, column_name)

            return_data[column_name] = column_data[idx]

            if self.loss_type == "contrastive":

                for example_type in ("positive", "negative"):

                    if example_idx is None:
                        indices = getattr(self, f"_{example_type}_indices")
                        example_idx = int(indices[idx, self.rng.integers(0, indices.dim)])

                    if example_idx != -1:
                        return_data[f"{column_name}_{example_type}"] = column_data[example_idx]
                    else:
                        fallback = getattr(self, f"_{column_name}_{example_type}_fallback")
                        return_data[f"{column_name}_{example_type}"] = fallback[idx]

            # if len(column_data) == 1:
            #     return_data[column_name] = column_data[0][idx]
            # elif self.loss_type == "default":
            #     if column_idx is None:
            #         column_idx = self.rng.integers(0, len(column_data))
            #     return_data[column_name] = column_data[column_idx][idx]
            # elif self.loss_type == "contrastive":
            #     if column_idx is None:
            #         column_idx = self.rng.integers(0, len(column_data))
            #     if row_idx is None:
            #         row_idx = self.rng.integers(0, len(self))
            #     return_data[column_name] = column_data[0][idx]
            #     return_data[f"{column_name}_positive"] = column_data[column_idx][idx]
            #     return_data[f"{column_name}_negative"] = column_data[0][row_idx]
            # else:
            #     raise NotImplementedError(f"Loss type '{self.loss_type}' not implemented.")

        return return_data

    def add_column(self, column: MemoryMappedDatasetColumn):
        if self.size is None:
            self.size = len(column)
        elif len(column) != len(self):
            raise ValueError(f"All columns must have the same number of examples. Column '{column.name}' has "
                             f"{len(column)} examples, expected {len(self)}.")
        if column.name in self.column_names:
            raise ValueError(f"Column '{column.name}' already exists in the dataset.")
        else:
            self.column_names.append(column.name)
            setattr(self, column.name, column)

    def rename_column(self, old_name: str, new_name: str):
        if old_name not in self.column_names:
            raise ValueError(f"Column '{old_name}' does not exist in the dataset.")
        if new_name in self.column_names:
            raise ValueError(f"Column '{new_name}' already exists in the dataset.")
        column_data = getattr(self, old_name)
        setattr(self, new_name, column_data)
        delattr(self, old_name)
        self.column_names.remove(old_name)
        self.column_names.append(new_name)
