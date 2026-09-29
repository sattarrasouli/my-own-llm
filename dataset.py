import torch
from torch.utils.data import Dataset


class LanguageModelDataset(Dataset):
    def __init__(
        self,
        token_ids: list[int],
        sequence_length: int,
    ):
        self.token_ids = token_ids
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.token_ids) - self.sequence_length

    def __getitem__(self, index):
        # Example:
        #
        # token_ids:
        # [10, 20, 30, 40, 50]
        #
        # sequence_length = 4
        #
        # input:
        #  [10, 20, 30, 40]
        #
        # target:
        #  [20, 30, 40, 50]

        input_tokens = self.token_ids[
            index : index + self.sequence_length
        ]

        target_tokens = self.token_ids[
            index + 1 : index + self.sequence_length + 1
        ]

        return (
            torch.tensor(input_tokens, dtype=torch.long),
            torch.tensor(target_tokens, dtype=torch.long),
        )