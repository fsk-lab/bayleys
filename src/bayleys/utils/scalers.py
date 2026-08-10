from abc import ABC, abstractmethod
import torch


class Scaler(ABC):
    """
    Implementation of a basic scaler interface for scaling the input and output data of probabilistic surrogate models
    in pytorch. Follows the general scikit-learn API, adding a further 'inverse_transform_variance' method for
    re-transforming the predicted variances of the surrogate model back to the original scale.

    Child classes must implement the 'fit', 'transform', 'inverse_transform' and 'inverse_transform_variance' methods.
    """

    device = torch.device("cpu")

    @abstractmethod
    def fit(self, x: torch.Tensor) -> None:
        """
        Fits the scaler to the provided data.

        Args:
            x: Tensor of input data (Shape: (N, D))
        """
        raise NotImplementedError

    @abstractmethod
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transforms the provided data.

        Args:
            x: Tensor of input data (Shape: (N, D))

        Returns:
            Tensor: Transformed data (Shape: (N, D))
        """
        raise NotImplementedError

    def fit_transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fits the scaler to the provided data, and transforms it.

        Args:
            x: Tensor of input data (Shape: (N, D))

        Returns:
            Tensor: Transformed data (Shape: (N, D))
        """
        self.fit(x)
        return self.transform(x)

    @abstractmethod
    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inverse-transforms the provided data.

        Args:
            x: Tensor of input data (Shape: (N, D))

        Returns:
            Tensor: Inverse-transformed data (Shape: (N, D))
        """
        raise NotImplementedError

    @abstractmethod
    def inverse_transform_variance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inverse-transforms the provided variances.

        Args:
            x: Tensor of input data (Shape: (N, D))

        Returns:
            Tensor: Inverse-transformed data (Shape: (N, D))
        """
        raise NotImplementedError

    @abstractmethod
    def to(self, device: torch.device) -> "Scaler":
        """
        Moves the scaler to the provided device.

        Args:
            device: Device to move the scaler to
        """
        raise NotImplementedError


class StandardScaler(Scaler):
    """
    Implementation of the standard scaler that removes the mean and scales to unit variance for each dimension of the
    input vector.

    z = (x - µ) / s

    where µ is the mean of the training samples, and s is the standard deviation of the training samples.
    """
    def __init__(self):
        self._mean = None
        self._std = None

    def fit(self, x: torch.Tensor) -> None:
        """
        Fits the scaler to the provided data.

        Args:
            x: Tensor of input data (Shape: (N, D))
        """
        self._mean = x.mean(dim=0)
        self._std = x.std(dim=0)

        self._std[self._std == 0] = 1
        self._std[torch.isnan(self._std)] = 1

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transforms the provided data.

        Args:
            x: Tensor of input data (Shape: (N, D))

        Returns:
            Tensor: Transformed data (Shape: (N, D))
        """
        return (x - self._mean) / self._std

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inverse-transforms the provided data.

        Args:
            x: Tensor of input data (Shape: (N, D))

        Returns:
            Tensor: Inverse-transformed data (Shape: (N, D))
        """
        return x * self._std + self._mean

    def inverse_transform_variance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inverse-transforms the provided variances.

        Args:
            x: Tensor of input data (Shape: (N, D))

        Returns:
            Tensor: Inverse-transformed data (Shape: (N, D))
        """
        return x * self._std.pow(2)

    def to(self, device: torch.device) -> "StandardScaler":
        """
        Moves the scaler to the provided device.

        Args:
            device: Device to move the scaler to
        """
        self.device = device
        if self._mean is not None:
            self._mean = self._mean.to(device)
        if self._std is not None:
            self._std = self._std.to(device)

        return self


class MinMaxScaler(Scaler):
    """
    Implementation of the min-max scaler that scales each dimension of the input vector to the range [0, 1].

    z = (x - min) / (max - min)

    where min is the minimum value of the training samples, and max is the maximum value of the training samples.
    """
    def __init__(self):
        self._min = None
        self._max = None

    def fit(self, x: torch.Tensor) -> None:
        """
        Fits the scaler to the provided data.

        Args:
            x: Tensor of input data (Shape: (N, D))
        """
        x = x.to(self.device)

        self._min = x.min(dim=0)[0].to(self.device)
        self._max = x.max(dim=0)[0].to(self.device)

        if torch.all(torch.eq(self._max, self._min)):
            self._min = torch.zeros_like(self._min).to(self.device)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transforms the provided data.

        Args:
            x: Tensor of input data (Shape: (N, D))

        Returns:
            Tensor: Transformed data (Shape: (N, D))
        """
        return (x - self._min) / (self._max - self._min)

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inverse-transforms the provided data.

        Args:
            x: Tensor of input data (Shape: (N, D))

        Returns:
            Tensor: Inverse-transformed data (Shape: (N, D))
        """
        return x * (self._max - self._min) + self._min

    def inverse_transform_variance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inverse-transforms the provided variances.

        Args:
            x: Tensor of input data (Shape: (N, D))

        Returns:
            Tensor: Inverse-transformed data (Shape: (N, D))
        """
        return x * (self._max - self._min).pow(2)

    def to(self, device: torch.device) -> "MinMaxScaler":
        """
        Moves the scaler to the provided device.

        Args:
            device: Device to move the scaler to
        """
        self.device = device
        if self._min is not None:
            self._min = self._min.to(device)
        if self._max is not None:
            self._max = self._max.to(device)

        return self
