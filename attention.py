import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention.

    Input:
        [batch_size, sequence_length, embedding_dim]

    Output:
        [batch_size, sequence_length, embedding_dim]
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        max_seq_len: int,
    ):
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_heads"
            )

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        # Instead of having separate Q/K/V layers for
        # every head, we calculate all of them at once.
        self.qkv = nn.Linear(
            embedding_dim,
            3 * embedding_dim,
        )

        # Combines the outputs of all attention heads.
        self.output_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
        )

        # Causal mask prevents tokens from seeing
        # future tokens.
        mask = torch.tril(
            torch.ones(
                max_seq_len,
                max_seq_len,
            )
        )

        self.register_buffer(
            "causal_mask",
            mask,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:
            [B, T, C]

        B = batch size
        T = sequence length
        C = embedding dimension
        """

        batch_size, seq_len, _ = x.shape

        # ------------------------------------------------
        # 1. Create Q, K, V
        # ------------------------------------------------

        qkv = self.qkv(x)

        # [B, T, 3C]
        #
        # Split into:
        #
        # [B, T, C]
        # [B, T, C]
        # [B, T, C]

        Q, K, V = qkv.chunk(3, dim=-1)

        # ------------------------------------------------
        # 2. Split into multiple heads
        # ------------------------------------------------

        # Current:
        #
        # [B, T, C]
        #
        # We want:
        #
        # [B, num_heads, T, head_dim]

        Q = Q.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        )

        K = K.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        )

        V = V.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        )

        # Move heads before sequence dimension.
        #
        # [B, T, H, D]
        #
        # becomes:
        #
        # [B, H, T, D]

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # ------------------------------------------------
        # 3. Calculate attention scores
        # ------------------------------------------------

        # Q:
        # [B, H, T, D]
        #
        # K.transpose:
        # [B, H, D, T]
        #
        # Result:
        # [B, H, T, T]

        scores = Q @ K.transpose(-2, -1)

        scores = scores / math.sqrt(
            self.head_dim
        )

        # ------------------------------------------------
        # 4. Causal masking
        # ------------------------------------------------

        mask = self.causal_mask[
            :seq_len,
            :seq_len,
        ]

        scores = scores.masked_fill(
            mask == 0,
            float("-inf"),
        )

        # ------------------------------------------------
        # 5. Softmax
        # ------------------------------------------------

        attention_weights = F.softmax(
            scores,
            dim=-1,
        )

        # ------------------------------------------------
        # 6. Weighted sum of Values
        # ------------------------------------------------

        output = attention_weights @ V

        # [B, H, T, D]

        # ------------------------------------------------
        # 7. Combine attention heads
        # ------------------------------------------------

        output = output.transpose(1, 2)

        # [B, T, H, D]

        output = output.contiguous().view(
            batch_size,
            seq_len,
            self.embedding_dim,
        )

        # ------------------------------------------------
        # 8. Final linear projection
        # ------------------------------------------------

        output = self.output_projection(output)

        return output