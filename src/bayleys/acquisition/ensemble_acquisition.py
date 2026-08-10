from typing import Optional
import time
import torch

from .base import Acquirer


class EnsembleAcquisition(Acquirer):
    """
    Acquisition strategy that selects candidates based on the acquisition function values averaged over an ensemble of
    acquisition functions with varying degrees of explorative behavior.
    """

    def __init__(
            self,
            acquisition_function: callable,
            acquisition_function_kwargs: dict,
            batch_size: int,
            max_exploration: float = 1.0,
    ):
        super().__init__(acquisition_function, acquisition_function_kwargs, batch_size)

        self.max_exploration = max_exploration
        self.exploration_values = torch.linspace(0, max_exploration, steps=batch_size).tolist()

    def acquire(self, mean: torch.Tensor, var: torch.Tensor, encodings: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Acquire candidates based on the acquisition function values from an ensemble of acquisition functions.

        Args:
            mean (torch.Tensor): Mean tensor of shape (N, 1)
            var (torch.Tensor): Variance tensor of shape (N, 1)
            encodings (torch.Tensor, optional): Feature encodings tensor of shape (N, D)

        Returns:
            torch.Tensor: Indices of acquired candidates of shape (batch_size,)
        """
        start_time = time.time()

        acquired_indices = []
        for exploration in self.exploration_values:
            self.acquisition_function_kwargs["exploration"] = exploration
            acquisition_values = self.acquisition_function(mean, var, **self.acquisition_function_kwargs).flatten()
            acquisition_values[acquired_indices] = - torch.inf
            acquired_indices.append(torch.argmax(acquisition_values).item())

        self.logger.debug(f"{self.__class__.__name__}: Acquired {self.batch_size} candidates using the "
                          f"{self.acquisition_function.__name__} acquisition function in {time.time() - start_time:.1f}"
                          f" seconds.")

        return torch.tensor(acquired_indices, dtype=torch.long)

    @property
    def name(self) -> str:
        """
        Returns the name of the acquisition strategy.
        """
        return f"EnsAcq-{self.max_exploration:.2f}-{self.acquisition_function.__name__}-{self.batch_size}"
