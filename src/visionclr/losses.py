
import torch
from torch import nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross-Entropy loss for SimCLR.

    Args:
        temperature: Controls the sharpness of the similarity distribution.

    Input:
        z1: [N, D], embeddings from the first augmented view
        z2: [N, D], embeddings from the second augmented view

    Returns:
        A scalar contrastive loss.
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()

        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.temperature = temperature

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> torch.Tensor:
        if z1.ndim != 2 or z2.ndim != 2:
            raise ValueError("Inputs must have shape [batch_size, embedding_dim]")

        if z1.shape != z2.shape:
            raise ValueError("z1 and z2 must have identical shapes")

        batch_size = z1.shape[0]

        if batch_size < 2:
            raise ValueError("Batch size must be at least 2")

        # Combine views: [z1_0, ..., z1_N-1, z2_0, ..., z2_N-1]
        z = torch.cat([z1, z2], dim=0)

        # Normalize so the dot product is cosine similarity.
        z = F.normalize(z, p=2, dim=1)

        # Similarity matrix: [2N, 2N]
        logits = torch.matmul(z, z.T) / self.temperature

        # The diagonal compares every embedding with itself.
        # Exclude those comparisons from the denominator.
        diagonal_mask = torch.eye(
            2 * batch_size,
            dtype=torch.bool,
            device=z.device,
        )
        logits = logits.masked_fill(diagonal_mask, float("-inf"))

        # Positive indices:
        # z1[i] pairs with z2[i], at index i + N.
        # z2[i] pairs with z1[i], at index i - N.
        targets = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=z.device),
            torch.arange(0, batch_size, device=z.device),
        ])

        # Each row is one anchor's classification problem.
        return F.cross_entropy(logits, targets)
