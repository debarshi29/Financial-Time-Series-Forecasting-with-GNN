"""Singleton FinBERT loader — downloads once, caches to HuggingFace default cache."""
from __future__ import annotations

import torch

_tokenizer = None
_model = None
_device = None


def get_finbert(device: str | None = None):
    global _tokenizer, _model, _device
    if _model is not None:
        return _tokenizer, _model

    from transformers import BertTokenizer, BertForSequenceClassification

    model_name = "ProsusAI/finbert"
    print(f"[FinBERT] Loading {model_name} ...")
    _tokenizer = BertTokenizer.from_pretrained(model_name)
    _model = BertForSequenceClassification.from_pretrained(model_name)
    _model.eval()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    _device = device
    _model = _model.to(_device)
    print(f"[FinBERT] Ready on {_device}")
    return _tokenizer, _model


def score_texts(texts: list[str], device: str | None = None, batch_size: int = 16) -> list[dict]:
    """Score a list of texts with FinBERT.
    Returns list of dicts: {positive, negative, neutral} probabilities.
    Labels are ordered by FinBERT config: 0=positive, 1=negative, 2=neutral.
    """
    if not texts:
        return []

    import torch.nn.functional as F
    tokenizer, model = get_finbert(device)
    dev = next(model.parameters()).device
    results = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {k: v.to(dev) for k, v in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()

        # FinBERT label order: positive=0, negative=1, neutral=2
        for row in probs:
            results.append({
                "positive": float(row[0]),
                "negative": float(row[1]),
                "neutral":  float(row[2]),
            })

    return results
