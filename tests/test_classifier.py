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