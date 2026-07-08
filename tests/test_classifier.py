"""
Tests for the PyTorch document-type classifier: vocabulary, model architecture,
training-data generation, and inference (including graceful degradation when
no checkpoint exists).

Run: pytest tests/test_classifier.py -v
"""

import sys
import os
import json
import tempfile
from pathlib import Path

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.ml.model import Vocabulary, DocTypeClassifier, collate_batch, tokenize, LABELS

# ─── Tokenization & Vocabulary ────────────────────────────────────────────────

def test_tokenize_lowercases_and_splits():
    tokens = tokenize("INVOICE #12345 — Due: $5,000.00")
    assert "invoice" in tokens
    assert "12345" in tokens
    assert "5" in tokens and "000" in tokens  # punctuation splits numbers, expected for this simple tokenizer


def test_tokenize_empty_string():
    assert tokenize("") == []

def test_vocabulary_build_respects_min_freq():
    texts = ["alpha alpha alpha", "beta beta", "gamma"]
    vocab = Vocabulary.build(texts, min_freq=2, max_size=100)
    # "alpha" (3x) and "beta" (2x) should be included; "gamma" (1x) should not
    assert "alpha" in vocab.token_to_id
    assert "beta" in vocab.token_to_id
    assert "gamma" not in vocab.token_to_id


def test_vocabulary_unk_fallback():
    vocab = Vocabulary.build(["known word"], min_freq=1, max_size=100)
    encoded = vocab.encode("totally unseen vocabulary here")
    # Every token is OOV, so every id should be the <unk> id
    unk_id = vocab.token_to_id[Vocabulary.UNK]
    assert all(idx == unk_id for idx in encoded)


def test_vocabulary_encode_truncates_to_max_len():
    vocab = Vocabulary.build(["word " * 500], min_freq=1, max_size=1000)
    encoded = vocab.encode("word " * 500, max_len=50)
    assert len(encoded) == 50

def test_vocabulary_encode_empty_text_returns_unk():
    vocab = Vocabulary.build(["some text"], min_freq=1)
    encoded = vocab.encode("")
    assert encoded == [vocab.token_to_id[Vocabulary.UNK]]


def test_vocabulary_save_and_load_roundtrip():
    vocab = Vocabulary.build(["alpha beta gamma delta"], min_freq=1, max_size=100)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "vocab.json"
        vocab.save(path)
        loaded = Vocabulary.load(path)
        assert loaded.token_to_id == vocab.token_to_id
        assert len(loaded) == len(vocab)

# ─── Model architecture ───────────────────────────────────────────────────────

def test_model_forward_shape():
    model = DocTypeClassifier(vocab_size=100, embed_dim=16, hidden_dim=8, n_classes=6)
    tokens = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    offsets = torch.tensor([0], dtype=torch.long)
    logits = model(tokens, offsets)
    assert logits.shape == (1, 6)  # batch size 1, 6 classes


def test_model_forward_batched():
    """Two documents of different lengths in one EmbeddingBag-style batch."""
    model = DocTypeClassifier(vocab_size=100, embed_dim=16, hidden_dim=8, n_classes=6)
    batch = [([1, 2, 3], 0), ([4, 5], 1)]
    tokens, offsets, labels = collate_batch(batch)
    logits = model(tokens, offsets)
    assert logits.shape == (2, 6)
    assert labels.tolist() == [0, 1]

def test_collate_batch_offsets_correct():
    """Offsets must mark the start index of each example in the flattened tensor."""
    batch = [([10, 20, 30], 0), ([40, 50], 1), ([60], 2)]
    tokens, offsets, labels = collate_batch(batch)
    assert tokens.tolist() == [10, 20, 30, 40, 50, 60]
    assert offsets.tolist() == [0, 3, 5]  # doc1 starts at 0, doc2 at 3, doc3 at 5


def test_model_is_trainable_single_step():
    """A single optimizer step should reduce loss on a trivially separable batch."""
    torch.manual_seed(0)
    model = DocTypeClassifier(vocab_size=50, embed_dim=8, hidden_dim=8, n_classes=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    criterion = torch.nn.CrossEntropyLoss()

    batch = [([1, 2, 3], 0), ([10, 11, 12], 1)] * 5
    tokens, offsets, labels = collate_batch(batch)

    losses = []
    for _ in range(20):
        optimizer.zero_grad()
        logits = model(tokens, offsets)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0]  # loss should decrease with training
