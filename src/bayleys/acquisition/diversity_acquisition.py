from typing import Callable, Optional, Literal
import time
import torch

from .base import Acquirer


class DiversityAcquisition(Acquirer):
    """
    Acquisition strategy that selects candidates based on both acquisition function values and diversity in the feature
    space. Acquires all data points sequentially, calculating a score based on the acquisition function value and the
    average distance to already acquired points.
    """

    def __init__(
            self,
            acquisition_function: Callable,
            acquisition_function_kwargs: dict,
            batch_size: int,
            metric: Literal["euclidean", "cosine"] = "euclidean",
            diversity_weight: float = 0.5,
    ):
        super().__init__(acquisition_function, acquisition_function_kwargs, batch_size)
        self.metric = metric
        self.diversity_weight = diversity_weight

    def acquire(self, mean: torch.Tensor, var: torch.Tensor, encodings: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Acquire candidates based on a combination of acquisition function values and diversity in the feature space.

        Args:
            mean (torch.Tensor): Mean tensor of shape (N, 1)
            var (torch.Tensor): Variance tensor of shape (N, 1)
            encodings (torch.Tensor, optional): Feature encodings tensor of shape (N, D)

        Returns:
            torch.Tensor: Indices of acquired candidates of shape (batch_size,)
        """
        start_time = time.time()

        if encodings is None:
            raise ValueError("Encodings must be provided for DiversityAcquisition.")

        acquisition_values = self.acquisition_function(mean, var, **self.acquisition_function_kwargs).flatten()
        min_val, max_val = acquisition_values.min(), acquisition_values.max()
        acquisition_values = (acquisition_values - min_val) / (max_val - min_val + 1e-8)

        all_distances = torch.full((encodings.shape[0], self.batch_size), torch.nan)
        acquired_indices = []

        for i in range(self.batch_size):

            if i == 0:
                distances = 0.0
            else:
                max_sim, min_sim = all_distances[:, :i].flatten().max(), all_distances[:, :i].flatten().min()
                distances = ((all_distances[:, :i] - min_sim) / (max_sim - min_sim + 1e-8)).mean(dim=1)

            scores = (1 - self.diversity_weight) * acquisition_values + self.diversity_weight * distances
            scores[acquired_indices] = - torch.inf
            acquired_index = torch.argmax(scores).item()
            acquired_indices.append(acquired_index)

            if i < self.batch_size - 1:
                new_encoding = encodings[acquired_index].unsqueeze(0)
                if self.metric == "euclidean":
                    dists = torch.cdist(encodings, new_encoding, p=2).squeeze(1)
                elif self.metric == "cosine":
                    dists = 1 - torch.nn.functional.cosine_similarity(encodings, new_encoding, dim=1)
                else:
                    raise ValueError(f"Unsupported distance metric: {self.metric}")

                all_distances[:, i] = dists

        self.logger.debug(f"{self.__class__.__name__}: Acquired {self.batch_size} candidates using the "
                          f"{self.acquisition_function.__name__} acquisition function in {time.time() - start_time:.1f}"
                          f" seconds.")

        return torch.tensor(acquired_indices, dtype=torch.long)

    @property
    def name(self) -> str:
        """
        Returns the name of the acquirer.

        Returns:
            str: Name of the acquirer.
        """
        return f"DivAcq-{self.diversity_weight:.2f}-{self.acquisition_function.__name__}-{self.batch_size}"
