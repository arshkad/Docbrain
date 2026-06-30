"""
Trains the DocBrain document-type classifier.

Usage:
    python ml_training/train.py

Reads ml_training/data/{train,val}.jsonl (generate with generate_data.py first),
trains a small EmbeddingBag classifier, prints per-epoch val accuracy, and saves:
    app/ml/checkpoints/classifier.pt   — model weights + config
    app/ml/checkpoints/vocab.json      — token vocabulary
"""

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ml.model import DocTypeClassifier, Vocabulary, collate_batch, LABELS, tokenize

DATA_DIR = Path(__file__).parent / "data"
CHECKPOINT_DIR = Path(__file__).parent.parent / "app" / "ml" / "checkpoints"


class DocDataset(Dataset):
    def __init__(self, path: Path, vocab: Vocabulary, label_to_id: dict[str, int]):
        self.examples = []
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                token_ids = vocab.encode(row["text"])
                self.examples.append((token_ids, label_to_id[row["label"]]))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def load_jsonl_texts(path: Path) -> list[str]:
    with open(path) as f:
        return [json.loads(line)["text"] for line in f]


def evaluate(model, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for tokens, offsets, labels in loader:
            tokens, offsets, labels = tokens.to(device), offsets.to(device), labels.to(device)
            logits = model(tokens, offsets)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total else 0.0


def main():
    train_path = DATA_DIR / "train.jsonl"
    val_path = DATA_DIR / "val.jsonl"

    if not train_path.exists():
        print("No training data found. Run `python ml_training/generate_data.py` first.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")