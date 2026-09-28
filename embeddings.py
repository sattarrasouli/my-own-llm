import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    Converts token IDs into dense vectors.

    Example:
        token IDs:
            [12, 45, 7]

        become:
            [
                [0.12, -0.31, ...],
                [0.42,  0.17, ...],
                [-0.08, 0.91, ...]
            ]
    """

    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embedding(tokens)


class PositionalEmbedding(nn.Module):
    """
    Gives each position in the sequence a learnable vector.

    Position 0 gets one vector,
    position 1 gets another,
    etc.
    """

    def __init__(self, max_seq_len: int, embedding_dim: int):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=max_seq_len,
            embedding_dim=embedding_dim,
        )

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        return self.embedding(positions)


class GPTEmbedding(nn.Module):
    """
    Combines token embeddings and positional embeddings.

        token embedding
              +
        positional embedding
              ↓
        Transformer input
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        embedding_dim: int,
    ):
        super().__init__()

        self.token_embedding = TokenEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
        )

        self.position_embedding = PositionalEmbedding(
            max_seq_len=max_seq_len,
            embedding_dim=embedding_dim,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:

        # tokens shape:
        # [batch_size, sequence_length]

        batch_size, sequence_length = tokens.shape

        token_vectors = self.token_embedding(tokens)

        # [sequence_length]
        positions = torch.arange(
            sequence_length,
            device=tokens.device,
        )

        position_vectors = self.position_embedding(
            positions
        )

        # Broadcasting adds position vectors
        # to every item in the batch.
        return token_vectors + position_vectors
