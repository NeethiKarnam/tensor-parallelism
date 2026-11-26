"""Tiny Shakespeare as fixed-length character windows, one-hot encoded in numpy.

The model sees the last `context` characters and predicts the next one. Inputs are
one-hot vectors concatenated over the window, shape (B, context * vocab), so the first
linear layer acts as a per-position embedding table and the model needs no nn.Embedding.

Every rank must draw the same batches: build the sampler with the same seed everywhere.
"""
import os
import urllib.request

import numpy as np

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "tinyshakespeare.txt")


def download(path=DEFAULT_PATH):
    """Download the text once; later calls reuse the file."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".part"
        urllib.request.urlretrieve(URL, tmp)
        os.replace(tmp, path)  # atomic: other ranks never see a half-written file
    return path


class CharDataset:
    def __init__(self, path=DEFAULT_PATH, context=8, val_frac=0.1):
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        self.chars = sorted(set(text))
        self.vocab = len(self.chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.context = context
        ids = np.array([self.stoi[c] for c in text], dtype=np.int64)
        split = int(len(ids) * (1 - val_frac))
        self.train, self.val = ids[:split], ids[split:]

    @property
    def input_dim(self):
        return self.context * self.vocab

    def encode(self, s):
        return np.array([self.stoi[c] for c in s], dtype=np.int64)

    def decode(self, ids):
        return "".join(self.chars[i] for i in ids)

    def one_hot(self, windows):
        """(B, context) int ids -> (B, context * vocab) float32."""
        return np.eye(self.vocab, dtype=np.float32)[windows].reshape(len(windows), -1)

    def batch(self, rng, batch_size, split="train"):
        """Random windows: X (B, context*vocab) float32, y (B,) next-char ids."""
        ids = self.train if split == "train" else self.val
        starts = rng.randint(0, len(ids) - self.context, size=batch_size)
        windows = ids[starts[:, None] + np.arange(self.context)]
        return self.one_hot(windows), ids[starts + self.context]
