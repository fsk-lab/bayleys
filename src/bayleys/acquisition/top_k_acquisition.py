from typing import Optional
import time
import torch

from .base import Acquirer


class TopKAcquisition(Acquirer):
    """
    Acquisition strategy that selects the top-k candidates based on the acquisition function values.
    """

    def acquire(self, mean: torch.Tensor, var: torch.Tensor, encodings: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Acquire the top-k candidates based on the acquisition function values.

        Args:
            mean (torch.Tensor): Mean tensor of shape (N, 1)
            var (torch.Tensor): Variance tensor of shape (N, 1)
            encodings (torch.Tensor, optional): Feature encodings tensor of shape (N, D)

        Returns:
            torch.Tensor: Indices of acquired candidates of shape (batch_size,)
        """
        start_time = time.time()

        acquisition_values = self.acquisition_function(mean, var, **self.acquisition_function_kwargs)
        _, indices = torch.topk(acquisition_values.flatten(), self.batch_size, largest=True, sorted=True)

        self.logger.debug(f"{self.__class__.__name__}: Acquired {self.batch_size} candidates using the "
                          f"{self.acquisition_function.__name__} acquisition function in {time.time() - start_time:.1f}"
                          f" seconds.")

        return indices

    @property
    def name(self) -> str:
        """
        Returns the name of the acquisition strategy.
        """
        return f"TopK-{self.acquisition_function.__name__}-{self.batch_size}"
