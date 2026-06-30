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
    
   # Build vocab from training text only (standard practice — val/test must
    # not leak into vocabulary construction)
    train_texts = load_jsonl_texts(train_path)
    vocab = Vocabulary.build(train_texts, min_freq=2, max_size=8000)
    print(f"Vocabulary size: {len(vocab)}")

    label_to_id = {label: i for i, label in enumerate(LABELS)}

    train_ds = DocDataset(train_path, vocab, label_to_id)
    val_ds = DocDataset(val_path, vocab, label_to_id)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_batch)

    model = DocTypeClassifier(vocab_size=len(vocab), embed_dim=64, hidden_dim=32, n_classes=len(LABELS)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    n_epochs = 12
    best_val_acc = 0.0
    start = time.time()

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0.0
        for tokens, offsets, labels in train_loader:
            tokens, offsets, labels = tokens.to(device), offsets.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(tokens, offsets)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            total_loss += loss.item()

        val_acc = evaluate(model, val_loader, device)
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch:2d}/{n_epochs} — loss: {avg_loss:.4f} — val_acc: {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "vocab_size": len(vocab),
                "embed_dim": 64,
                "hidden_dim": 32,
                "labels": LABELS,
                "val_acc": val_acc,
            }, CHECKPOINT_DIR / "classifier.pt")
            vocab.save(CHECKPOINT_DIR / "vocab.json")

    elapsed = time.time() - start
    print(f"\nTraining complete in {elapsed:.1f}s. Best val accuracy: {best_val_acc:.3f}")
    print(f"Checkpoint saved to {CHECKPOINT_DIR}/classifier.pt")


if __name__ == "__main__":
    main()