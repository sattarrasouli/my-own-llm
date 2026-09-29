import torch
import torch.nn as nn

from embedding import GPTEmbedding
from transformer import TransformerBlock


class MiniGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        embedding_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        # 1. Token + positional embeddings
        self.embeddings = GPTEmbedding(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            embedding_dim=embedding_dim,
        )

        # 2. Stack multiple Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    max_seq_len=max_seq_len,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # 3. Final normalization
        self.final_layer_norm = nn.LayerNorm(embedding_dim)

        # 4. Convert hidden representations → vocabulary logits
        self.lm_head = nn.Linear(
            embedding_dim,
            vocab_size,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens shape:
            [batch_size, sequence_length]

        returns:
            logits shape:
            [batch_size, sequence_length, vocab_size]
        """

        batch_size, sequence_length = tokens.shape

        if sequence_length > self.max_seq_len:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds "
                f"maximum sequence length {self.max_seq_len}"
            )

        # Token + positional embeddings
        x = self.embeddings(tokens)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final normalization
        x = self.final_layer_norm(x)

        # Vocabulary logits
        logits = self.lm_head(x)

        return logits