import torch
from torch.distributions import Normal
from torch import Tensor


def random(mean: Tensor, variance: Tensor, **kwargs) -> Tensor:
    """
    Random acquisition function that returns random values.

    Args:
        mean: Tensor of shape (N, ...)
        variance: Tensor of shape (N, ...)

    Returns:
        Tensor: Random tensor of shape (N, ...)
    """
    return torch.rand_like(mean)


def ucb(mean: Tensor, variance: Tensor, factor: float, **kwargs) -> Tensor:
    """
    Single objective upper confidence bound for a fixed set of independent points (each characterized by a mean and a
    variance).

    Args:
        mean: Tensor of shape (N, ...)
        variance: Tensor of shape (N, ...)
        factor: 1.0 for maximization, -1.0 for minimization

    Returns:
        Tensor: Upper confidence bound tensor of shape (N, ...)
    """
    exploration = kwargs.get("exploration", 0.2)

    return mean * factor + exploration * torch.sqrt(variance)


def ei(mean: Tensor, variance: Tensor, best_f: float, factor: float, **kwargs) -> Tensor:
    """
    Single objective expected improvement for a fixed set of independent points (each characterized by a mean and a
    variance).

    Args:
        mean: Tensor of shape (N, ...)
        variance: Tensor of shape (N, ...)
        best_f: Current best function value
        factor: 1.0 for maximization, -1.0 for minimization

    Returns:
        Tensor: Expected improvement tensor of shape (N, ...)
    """
    exploration = kwargs.get("exploration", 0.1)
    best_f = best_f * factor
    mean = mean * factor

    sigma = torch.sqrt(variance.clamp(min=1e-12))

    # handle the specific case where sigma is zero
    if torch.all(torch.isclose(sigma, torch.tensor(0.0), atol=1e-5)):
        improvement = mean - best_f - exploration
        if torch.any(improvement > 0):
            return torch.clamp(improvement, min=0.0)
        else:
            return improvement

    z = (mean - best_f - exploration) / sigma
    ei_vals = z * sigma * Normal(0, 1).cdf(z) + sigma * Normal(0, 1).log_prob(z).exp()

    return ei_vals.clamp(min=0.0)


def logEI(mean: Tensor, variance: Tensor, best_f: float, factor: float, **kwargs) -> Tensor:
    """
    Single objective log expected improvement for a fixed set of independent points (each characterized by a mean and a
    variance).

    Args:
        mean: Tensor of shape (N, ...)
        variance: Tensor of shape (N, ...)
        best_f: Current best function value
        factor: 1.0 for maximization, -1.0 for minimization

    Returns:
        Tensor: Log expected improvement tensor of shape (N, ...)
    """
    ei_vals = ei(mean, variance, best_f, factor, **kwargs).clamp(min=1e-12)
    return torch.log(ei_vals)

    # # Use the log-sum-exp trick for numerical stability
    # log_phi, log_Phi = Normal(0, 1).log_prob(z), torch.log(Normal(0, 1).cdf(z).clamp_min(1e-9))
    # log_term_1 = torch.log(u.clamp_min(0) + 1e-9) + log_Phi
    # log_term_2 = torch.log(sigma + 1e-9) + log_phi
    # max_term = torch.max(log_term_1, log_term_2)
    #
    # log_ei_vals = max_term + torch.log(torch.exp(log_term_1 - max_term) + torch.exp(log_term_2 - max_term))
    #
    # zero_std_mask = torch.isclose(sigma, torch.tensor(0.0))
    # log_ei_vals[zero_std_mask] = torch.log(torch.clamp(mean[zero_std_mask] - best_f - exploration, min=1e-9))
    #
    # return log_ei_vals
