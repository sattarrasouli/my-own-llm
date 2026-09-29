import torch

from tokenizer import CharacterTokenizer
from dataset import LanguageModelDataset
from model import MiniGPT
from trainer import Trainer


text = """
hello world
hello world
hello world
the cat sat on the mat
the cat sat on the mat
the dog ran outside
the dog ran outside
"""


# -------------------------
# 1. Tokenizer
# -------------------------

tokenizer = CharacterTokenizer(text)

token_ids = tokenizer.encode(text)

print("Vocabulary size:", tokenizer.vocab_size)


# -------------------------
# 2. Dataset
# -------------------------

sequence_length = 32

dataset = LanguageModelDataset(
    token_ids=token_ids,
    sequence_length=sequence_length,
)

print("Dataset size:", len(dataset))


# -------------------------
# 3. Model
# -------------------------

model = MiniGPT(
    vocab_size=tokenizer.vocab_size,
    max_seq_len=sequence_length,
    embedding_dim=128,
    num_layers=2,
    num_heads=4,
    dropout=0.1,
)


# -------------------------
# 4. Trainer
# -------------------------

trainer = Trainer(
    model=model,
    train_dataset=dataset,
    batch_size=4,
    learning_rate=3e-4,
)


# -------------------------
# 5. Training
# -------------------------

for epoch in range(50):

    loss = trainer.train_epoch()

    print(
        f"Epoch {epoch + 1:03d} | "
        f"Loss: {loss:.4f}"
    )