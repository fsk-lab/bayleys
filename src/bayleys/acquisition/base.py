from typing import Callable, Optional
from abc import ABC, abstractmethod
from logging import getLogger
import torch

from ..molecule_library import MoleculeLibrary


class Acquirer(ABC):
    """
    Base class for all acquisition function optimization strategies that select a batch of `k` candidates from a virtual
    molecular library based on the predicted mean and variance from a surrogate model.
    """

    def __init__(
            self,
            acquisition_function: Callable,
            acquisition_function_kwargs: dict,
            batch_size: int,
            **kwargs,
    ):

        self.acquisition_function = acquisition_function
        self.acquisition_function_kwargs = acquisition_function_kwargs

        self.batch_size = batch_size

        self.logger = getLogger("bayleys")

    def update(self, previous_acquisitions: MoleculeLibrary):
        """
        Update the acquirer with previously acquired candidates. This is useful for acquisition functions that depend
        on the previously acquired candidates, e.g. to update the best observed value for EI/LogEI/PI acquisition
        functions.

        Args:
            previous_acquisitions (MoleculeLibrary): Previously acquired candidates.
        """
        best_observed_value = torch.max(previous_acquisitions.labels)
        self.acquisition_function_kwargs["best_f"] = best_observed_value.item()
        self.acquisition_function_kwargs["factor"] = previous_acquisitions.factor  # 1.0 for max., -1.0 for min.

    @abstractmethod
    def acquire(self, mean: torch.Tensor, var: torch.Tensor, encodings: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Abstract method to be implemented by subclasses for acquiring candidates.

        Args:
            mean (torch.Tensor): Mean tensor of shape (N, 1)
            var (torch.Tensor): Variance tensor of shape (N, 1)
            encodings (torch.Tensor, optional): Feature encodings tensor of shape (N, D)

        Returns:
            torch.Tensor: Indices of acquired candidates of shape (batch_size,)
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Returns the name of the acquirer.

        Returns:
            str: Name of the acquirer.
        """
        raise NotImplementedError
