class CharacterTokenizer:
    def __init__(self, text: str):
        # Find every unique character in the dataset
        characters = sorted(set(text))

        # Special token for unknown characters
        self.unk_token = "<UNK>"

        # Create vocabulary
        self.vocab = [self.unk_token] + characters

        # character -> integer ID
        self.token_to_id = {
            token: index
            for index, token in enumerate(self.vocab)
        }

        # integer ID -> character
        self.id_to_token = {
            index: token
            for index, token in enumerate(self.vocab)
        }

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> list[int]:
        """
        Convert text into token IDs.
        """

        return [
            self.token_to_id.get(
                character,
                self.token_to_id[self.unk_token],
            )
            for character in text
        ]

    def decode(self, token_ids: list[int]) -> str:
        """
        Convert token IDs back into text.
        """

        return "".join(
            self.id_to_token[token_id]
            for token_id in token_ids
        )