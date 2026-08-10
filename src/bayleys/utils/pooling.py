from typing import Optional, Literal
import torch


def pool_transformer_outputs(
        pooling_method: Literal["cls", "mean", "max"],
        hidden_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Pools the outputs of a transformer model using the specified pooling_operation method.

    Args:
        pooling_method (str): The pooling_operation method to use ('cls', 'mean' or 'max').
        hidden_state (torch.Tensor): The last hidden state from the transformer model.
        attention_mask (Optional[torch.Tensor]): The attention mask used during tokenization.

    Returns:
        torch.Tensor: The pooled embeddings.
    """
    if pooling_method == "cls":
        embeddings = hidden_state[:, 0, :].contiguous()
    elif pooling_method == "mean":
        if attention_mask is None:
            raise ValueError("Attention mask is required for mean pooling_operation.")
        attention_mask = attention_mask.unsqueeze(-1).to(hidden_state.dtype)
        sum_embeddings = torch.sum(hidden_state * attention_mask, dim=1)
        denominator = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
        embeddings = sum_embeddings / denominator
    elif pooling_method == "max":
        if attention_mask is None:
            raise ValueError("Attention mask is required for max pooling_operation.")
        attention_mask = attention_mask.unsqueeze(-1).bool()
        hidden_state = hidden_state.masked_fill(~attention_mask, float('-inf'))
        embeddings = hidden_state.max(dim=1).values
    else:
        raise ValueError(f"Unsupported pooling_operation method: {pooling_method}")

    return embeddings
