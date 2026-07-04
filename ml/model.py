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

class DocTypeClassifier(nn.Module):
    """
    EmbeddingBag (mean-pooled bag-of-words) -> hidden MLP -> class logits.

    EmbeddingBag is used instead of Embedding + manual mean so variable-length
    documents can be batched efficiently via offsets, without padding to a
    fixed sequence length.
    """

    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_dim: int = 32, n_classes: int = 6):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, mode="mean")
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_classes),
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.embedding.weight, -0.5, 0.5)
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, token_ids: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(token_ids, offsets)
        return self.classifier(embedded)


def collate_batch(batch: list[tuple[list[int], int]]):
    """
    Flattens a batch of (token_ids, label) pairs into the concatenated
    token tensor + offsets tensor that EmbeddingBag expects.
    """
    labels = torch.tensor([label for _, label in batch], dtype=torch.long)
    token_lists = [torch.tensor(tokens, dtype=torch.long) for tokens, _ in batch]
    offsets = torch.tensor([0] + [len(t) for t in token_lists[:-1]], dtype=torch.long).cumsum(dim=0)
    flat_tokens = torch.cat(token_lists)
    return flat_tokens, offsets, labels

