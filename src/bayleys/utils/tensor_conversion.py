import numpy as np
import torch
import pyarrow as pa


def tensor_to_pyarrow(tensor: torch.Tensor) -> pa.FixedSizeListArray:
    """
    Converts a 2D PyTorch tensor to a pyarrow `FixedSizeListArray`.

    Args:
        tensor (torch.Tensor): A 2D PyTorch tensor of shape (num_rows, num_cols).

    Returns:
        pa.FixedSizeListArray: A pyarrow array representing the input tensor.
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("Input must be a PyTorch tensor.")
    if tensor.ndim != 2:
        raise ValueError("Input tensor must be 2D.")

    tensor = tensor.detach()

    if tensor.device.type != "cpu":
        tensor = tensor.cpu()

    if tensor.dtype != torch.float32:
        tensor = tensor.float()

    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    arr = tensor.numpy()
    num_cols = arr.shape[1]
    values = pa.array(arr.ravel(), type=pa.float32())

    return pa.FixedSizeListArray.from_arrays(values, num_cols)
