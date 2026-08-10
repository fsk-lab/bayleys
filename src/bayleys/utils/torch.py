import torch


def reset_weights(module: torch.nn.Module):
    """
    Resets the weights of a given PyTorch module. This function is useful for reinitializing the weights of a model
    before retraining.

    Args:
        module (torch.nn.Module): The PyTorch module whose weights are to be reset.
    """
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()