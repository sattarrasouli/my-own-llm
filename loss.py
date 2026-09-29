import torch
import torch.nn.functional as F


def language_model_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate next-token prediction loss.

    logits:
        [batch_size, sequence_length, vocab_size]

    targets:
        [batch_size, sequence_length]
    """

    batch_size, sequence_length, vocab_size = logits.shape

    # Flatten logits:
    #
    # [B, T, V]
    #      ↓
    # [B*T, V]
    logits = logits.reshape(
        batch_size * sequence_length,
        vocab_size,
    )

    # Flatten targets:
    #
    # [B, T]
    #    ↓
    # [B*T]
    targets = targets.reshape(
        batch_size * sequence_length
    )

    loss = F.cross_entropy(
        logits,
        targets,
    )

    return loss