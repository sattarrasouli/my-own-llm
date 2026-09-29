import torch
import torch.nn as nn

from myllm.attention import MultiHeadSelfAttention


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.

    Every token representation goes through
    the same small neural network independently.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int | None = None,
    ):
        super().__init__()

        # GPT-style Transformers commonly use
        # approximately 4x expansion.
        if hidden_dim is None:
            hidden_dim = 4 * embedding_dim

        self.network = nn.Sequential(
            nn.Linear(
                embedding_dim,
                hidden_dim,
            ),

            nn.GELU(),

            nn.Linear(
                hidden_dim,
                embedding_dim,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.network(x)


class TransformerBlock(nn.Module):
    """
    One complete GPT-style Transformer block.

    Architecture:

        Input
          │
          ▼
      LayerNorm
          │
          ▼
    Self-Attention
          │
          ▼
     Residual Add
          │
          ▼
      LayerNorm
          │
          ▼
       FeedForward
          │
          ▼
     Residual Add
          │
          ▼
        Output
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        max_seq_len: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(
            embedding_dim
        )

        self.attention = MultiHeadSelfAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
        )

        self.layer_norm_2 = nn.LayerNorm(
            embedding_dim
        )

        self.feed_forward = FeedForward(
            embedding_dim=embedding_dim,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        # ---------------------------------------------
        # Attention + Residual Connection
        # ---------------------------------------------

        residual = x

        x = self.layer_norm_1(x)

        x = self.attention(x)

        x = self.dropout(x)

        x = x + residual

        # ---------------------------------------------
        # Feed Forward + Residual Connection
        # ---------------------------------------------

        residual = x

        x = self.layer_norm_2(x)

        x = self.feed_forward(x)

        x = self.dropout(x)

        x = x + residual

        return x