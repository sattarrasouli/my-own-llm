import torch
from torch.utils.data import DataLoader

from loss import language_model_loss


class Trainer:

    def __init__(
        self,
        model,
        train_dataset,
        batch_size: int = 32,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
    ):
        self.model = model

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    def train_epoch(self):
        self.model.train()

        total_loss = 0.0

        for inputs, targets in self.train_loader:

            # Clear old gradients
            self.optimizer.zero_grad()

            # Forward pass
            logits = self.model(inputs)

            # Calculate loss
            loss = language_model_loss(
                logits,
                targets,
            )

            # Backpropagation
            loss.backward()

            # Update model parameters
            self.optimizer.step()

            total_loss += loss.item()

        average_loss = (
            total_loss / len(self.train_loader)
        )

        return average_loss