import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """
    Single-head causal self-attention.

    The model can attend to the current token and
    previous tokens, but never future tokens.
    """

    def __init__(self, embedding_dim: int, max_seq_len: int):
        super().__init__()

        self.embedding_dim = embedding_dim

        # Convert input embeddings into:
        # Query, Key, and Value
        self.query = nn.Linear(
            embedding_dim,
            embedding_dim,
        )

        self.key = nn.Linear(
            embedding_dim,
            embedding_dim,
        )

        self.value = nn.Linear(
            embedding_dim,
            embedding_dim,
        )

        # Causal mask:
        #
        # [[1, 0, 0, 0],
        #  [1, 1, 0, 0],
        #  [1, 1, 1, 0],
        #  [1, 1, 1, 1]]
        #
        # This prevents tokens from seeing the future.
        mask = torch.tril(
            torch.ones(
                max_seq_len,
                max_seq_len,
            )
        )

        # register_buffer means:
        # - it is part of the model
        # - it moves to CPU/GPU with the model
        # - it isn't a trainable parameter
        self.register_buffer(
            "causal_mask",
            mask,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape:

        [batch_size, sequence_length, embedding_dim]
        """

        batch_size, seq_len, _ = x.shape

        # ------------------------------------------------
        # 1. Create Query, Key, Value
        # ------------------------------------------------

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # ------------------------------------------------
        # 2. Calculate attention scores
        # ------------------------------------------------

        # K.transpose:
        #
        # [batch, seq, dim]
        #
        # becomes:
        #
        # [batch, dim, seq]

        scores = Q @ K.transpose(-2, -1)

        # ------------------------------------------------
        # 3. Scale the scores
        # ------------------------------------------------

        scores = scores / math.sqrt(
            self.embedding_dim
        )

        # ------------------------------------------------
        # 4. Apply causal mask
        # ------------------------------------------------

        mask = self.causal_mask[:seq_len, :seq_len]

        scores = scores.masked_fill(
            mask == 0,
            float("-inf"),
        )

        # ------------------------------------------------
        # 5. Convert scores to probabilities
        # ------------------------------------------------

        attention_weights = F.softmax(
            scores,
            dim=-1,
        )

        # ------------------------------------------------
        # 6. Weighted sum of Values
        # ------------------------------------------------

        output = attention_weights @ V

        return output
