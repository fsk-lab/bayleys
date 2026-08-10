from __future__ import annotations
from typing import Union, Tuple, List, Any
import numpy as np
import torch
from copy import copy


class NonTensorialData(object):
    """
    Data structure that can be used in place of a Tensor to be passed to DataLoaders and other PyTorch utilities.

    Implements the following important Tensor-like methods:
    - shape and __len__
    - __getitem__ (for integer indices, slices and boolean masks)
    - append()
    - __iter__
    - __deepcopy__
    - to()

    The underlying data can be a list of any type (e.g. networkx graphs, torch_geometric Data objects, etc.).
    """
    def __init__(self, data: List[Any]):
        self._data = data

    def __len__(self):
        return len(self._data)

    @property
    def shape(self) -> Tuple[int]:
        return len(self),

    def __iter__(self):
        return iter(self._data)

    def __str__(self):
        return str(self._data)

    def __getitem__(self, idx):

        if isinstance(idx, (int, slice)):
            return self._data[idx]

        elif isinstance(idx, (list, tuple, np.ndarray, torch.Tensor)):

            # convert boolean mask to indices
            if isinstance(idx, np.ndarray) and idx.dtype == np.bool:
                idx = np.where(idx)[0]
            elif isinstance(idx, torch.Tensor) and idx.dtype == torch.bool:
                idx = torch.where(idx)[0]

            return NonTensorialData([self._data[i] for i in idx])

        else:
            raise TypeError(f"Index type {type(idx)} not supported")

    def append(self, item: Union[list, NonTensorialData]):
        if isinstance(item, NonTensorialData):
            self._data.extend(item._data)
        elif isinstance(item, list):
            self._data.extend(item)

    def __deepcopy__(self, memo):
        return NonTensorialData(copy(self._data))

    def to(self, device, *args, **kwargs):
        """
        Dummy method to allow NonTensorialData to be handled like tensors. Does not actually do anything.
        """
        return self

    def detach(self):
        """
        Dummy method to allow NonTensorialData to be handled like tensors. Does not actually do anything.
        """
        return self

    def cpu(self):
        """
        Dummy method to allow NonTensorialData to be handled like tensors. Does not actually do anything.
        """
        return self

    def type(self, dtype):
        """
        Dummy method to allow NonTensorialData to be handled like tensors. Does not actually do anything.
        """
        return self

    @staticmethod
    def is_floating_point():
        return False

    @staticmethod
    def is_complex():
        return False


TensorLike = Union[torch.Tensor, NonTensorialData]
