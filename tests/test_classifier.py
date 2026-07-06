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