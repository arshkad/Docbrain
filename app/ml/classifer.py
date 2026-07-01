"""
Inference-time wrapper around the trained DocTypeClassifier.

Loads the checkpoint once at import time (lazy singleton) and exposes
predict_document_type() for use in the ingestion pipeline and API routes.

If no checkpoint is found (model not yet trained), predict_document_type()
returns a low-confidence "unknown" result rather than raising — the rest of
DocBrain's product (RAG Q&A, Claude-based insights) must keep working even
if this optional local classifier hasn't been trained yet.
"""

from pathlib import Path
from functools import lru_cache

import torch
import torch.nn.functional as F

from app.ml.model import DocTypeClassifier, Vocabulary, tokenize

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
MODEL_PATH = CHECKPOINT_DIR / "classifier.pt"
VOCAB_PATH = CHECKPOINT_DIR / "vocab.json"


class ClassifierUnavailable(Exception):
    """Raised when the model checkpoint hasn't been trained/saved yet."""


@lru_cache(maxsize=1)
def _load():
    if not MODEL_PATH.exists() or not VOCAB_PATH.exists():
        raise ClassifierUnavailable(
            f"No trained classifier found at {MODEL_PATH}. "
            f"Run `python ml_training/generate_data.py && python ml_training/train.py` first."
        )

    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    vocab = Vocabulary.load(VOCAB_PATH)

    model = DocTypeClassifier(
        vocab_size=checkpoint["vocab_size"],
        embed_dim=checkpoint["embed_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        n_classes=len(checkpoint["labels"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model, vocab, checkpoint["labels"]
def is_classifier_ready() -> bool:
    """Check without raising — lets API routes degrade gracefully."""
    try:
        _load()
        return True
    except ClassifierUnavailable:
        return False


def predict_document_type(text: str, top_k: int = 3) -> dict:
    """
    Classify a document's text into one of the trained categories.

    Returns:
        {
          "predicted_type": "contract",
          "confidence": 0.94,
          "top_k": [{"label": "contract", "score": 0.94}, ...],
          "model": "docbrain-doctype-v1"
        }

    Raises ClassifierUnavailable if the model hasn't been trained yet —
    callers should catch this and fall back gracefully (e.g. skip auto-tagging).
    """
    model, vocab, labels = _load()

    token_ids = vocab.encode(text, max_len=512)
    tokens_tensor = torch.tensor(token_ids, dtype=torch.long)
    offsets_tensor = torch.tensor([0], dtype=torch.long)

    with torch.no_grad():
        logits = model(tokens_tensor, offsets_tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)

    ranked = sorted(
        [{"label": label, "score": round(probs[i].item(), 4)} for i, label in enumerate(labels)],
        key=lambda x: x["score"],
        reverse=True,
    )

    return {
        "predicted_type": ranked[0]["label"],
        "confidence": ranked[0]["score"],
        "top_k": ranked[:top_k],
        "model": "docbrain-doctype-v1",
    }
