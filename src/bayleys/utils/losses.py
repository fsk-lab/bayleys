import torch


def masked_language_modeling_loss(
        mlm_logits: torch.Tensor,
        labels: torch.Tensor,
        ignore_index: int = -100
):
    """
    Computes the token prediction loss for masked language modeling (MLM) as a cross-entropy loss.

    Args:
        mlm_logits (torch.Tensor): The predicted token logits from the model. Shape (batch_size, seq_len, vocab_size)
        labels (torch.Tensor): The true token labels for the masked positions. Shape (batch_size, seq_len).
        ignore_index (int): The index in `labels` that indicates positions to be ignored in the loss computation.

    Returns:
        torch.Tensor: The computed token prediction loss as a scalar tensor.
    """
    vocab_size = mlm_logits.shape[-1]

    return torch.nn.functional.cross_entropy(
        mlm_logits.view(-1, vocab_size),
        labels.view(-1),
        ignore_index=ignore_index
    )


def contrastive_loss(
        anchors: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor,
        temperature: float = 0.05
) -> torch.Tensor:
    """
    Computes a multiple-negatives ranking loss for SMILES triples: (Anchor, Positive, Negatives). The loss encourages
    the model to assign higher similarity scores to the positive pairs (Anchor, Positive) than to the negative pairs
    (Anchor, Negatives).

    Args:
        anchors (torch.Tensor): Tensor of shape (batch_size, embedding_dim) representing the anchor embeddings.
        positives (torch.Tensor): Tensor of shape (batch_size, embedding_dim) representing the positive embeddings.
        negatives (torch.Tensor): Tensor of shape (batch_size, embedding_dim) representing the negative
                                  embeddings.
        temperature (float): Temperature scaling factor for the similarity scores.

    Returns:
        torch.Tensor: The computed contrastive loss as a scalar tensor.
    """
    anchors = torch.nn.functional.normalize(anchors, p=2, dim=-1)
    positives = torch.nn.functional.normalize(positives, p=2, dim=-1)
    negatives = torch.nn.functional.normalize(negatives, p=2, dim=-1)

    positive_scores = anchors @ positives.T / temperature
    negative_scores = anchors @ negatives.T / temperature

    labels = torch.arange(anchors.size(0)).to(anchors.device)
    logits = torch.cat([positive_scores, negative_scores], dim=1)

    return torch.nn.functional.cross_entropy(logits, labels)