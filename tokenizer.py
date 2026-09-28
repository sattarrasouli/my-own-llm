class CharacterTokenizer:

    def __init__(self, text):
        chars = sorted(set(text))

        self.stoi = {
            ch: i
            for i, ch in enumerate(chars)
        }

        self.itos = {
            i: ch
            for ch, i in self.stoi.items()
        }

    def encode(self, text):
        return [
            self.stoi[ch]
            for ch in text
        ]

    def decode(self, tokens):
        return "".join(
            self.itos[token]
            for token in tokens
        )

    @property
    def vocab_size(self):
        return len(self.stoi)
