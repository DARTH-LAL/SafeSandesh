from __future__ import annotations

import re
import unicodedata
from typing import Dict, Iterable, List, Sequence

import numpy as np

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except Exception:  # pragma: no cover - optional dependency guard
    torch = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None


DEFAULT_INDICBERT_NAME = "ai4bharat/IndicBERTv2-MLM-only"
DEFAULT_TRANSFORMER_NAME = DEFAULT_INDICBERT_NAME
WHITESPACE_RE = re.compile(r"\s+", flags=re.UNICODE)


def normalize_text(text: object) -> str:
    if text is None:
        return ""
    out = unicodedata.normalize("NFKC", str(text))
    out = out.replace("\u200b", " ").replace("\ufeff", " ")
    return WHITESPACE_RE.sub(" ", out).strip()


def normalize_texts(texts: Iterable[object]) -> List[str]:
    return [normalize_text(text) for text in texts]


def tokenize_texts(tokenizer, texts: Sequence[object], max_length: int):
    clean = normalize_texts(texts)
    return tokenizer(
        clean,
        truncation=True,
        padding="max_length",
        max_length=int(max_length),
        return_attention_mask=True,
        return_token_type_ids=False,
        return_tensors="pt",
    )


def batch_predict_logits(
    model,
    tokenizer,
    texts: Sequence[object],
    max_length: int,
    batch_size: int,
    device,
) -> np.ndarray:
    if torch is None:
        raise RuntimeError("PyTorch is required for transformer inference.")

    model.eval()
    outputs: List[np.ndarray] = []
    clean_texts = normalize_texts(texts)

    for start in range(0, len(clean_texts), batch_size):
        batch_texts = clean_texts[start : start + batch_size]
        enc = tokenizer(
            batch_texts,
            truncation=True,
            padding="max_length",
            max_length=int(max_length),
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        outputs.append(logits.detach().cpu().numpy())

    if not outputs:
        return np.empty((0, 0), dtype=np.float32)
    return np.vstack(outputs)
