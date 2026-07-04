"""
PyTorch document-type classifier.

Architecture: a bag-of-words embedding classifier.
  text -> tokenize -> vocab indices -> EmbeddingBag (mean-pooled) -> MLP -> logits

This is intentionally lightweight (no transformer, no GPU needed) because:
  - DocBrain already gets rich semantic understanding from Claude + ChromaDB
    embeddings for retrieval. This classifier's job is narrow: fast, cheap,
    local document-type tagging at upload time, without an API call.
  - EmbeddingBag + mean pooling is a strong, well-known baseline for short-to-
    medium document classification and trains in seconds on CPU.

The vocabulary and label list are persisted alongside model weights so
inference doesn't depend on retraining artifacts being in sync by accident.
"""

import re
import json
from pathlib import Path

import torch
import torch.nn as nn


LABELS = ["contract", "invoice", "report", "policy", "memo", "other"]
TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase + alphanumeric tokenization. Simple on purpose — the model
    learns which tokens matter, we don't need stemming/lemmatization here."""
    return TOKEN_RE.findall(text.lower())


class Vocabulary:
    """Maps tokens <-> integer indices, with an <unk> bucket for unseen tokens."""

    UNK = "<unk>"
    PAD = "<pad>"

    def __init__(self, tokens_to_ids: dict[str, int] | None = None):
        self.token_to_id = tokens_to_ids or {self.PAD: 0, self.UNK: 1}

    @classmethod
    def build(cls, texts: list[str], min_freq: int = 2, max_size: int = 8000) -> "Vocabulary":
        from collections import Counter
        counter = Counter()
        for text in texts:
            counter.update(tokenize(text))

        vocab = cls()
        next_id = len(vocab.token_to_id)
        for token, freq in counter.most_common(max_size):
            if freq < min_freq:
                continue
            if token not in vocab.token_to_id:
                vocab.token_to_id[token] = next_id
                next_id += 1
        return vocab
        
    def encode(self, text: str, max_len: int = 256) -> list[int]:
        ids = [self.token_to_id.get(tok, self.token_to_id[self.UNK]) for tok in tokenize(text)]
        return ids[:max_len] if ids else [self.token_to_id[self.UNK]]

    def __len__(self):
        return len(self.token_to_id)

    def save(self, path: Path):
        with open(path, "w") as f:
            json.dump(self.token_to_id, f)

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        with open(path) as f:
            return cls(json.load(f))
