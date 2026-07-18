from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

try:
    import torch
    from torch import nn
    from torch.nn.utils.rnn import pack_padded_sequence
except Exception:  # pragma: no cover - optional dependency guard
    torch = None
    nn = None
    pack_padded_sequence = None


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def normalize_text(text: object) -> str:
    if text is None:
        return ""
    out = unicodedata.normalize("NFKC", str(text))
    out = out.replace("\u200b", " ").replace("\ufeff", " ")
    out = re.sub(r"\s+", " ", out).strip().lower()
    return out


def tokenize(text: object) -> List[str]:
    clean = normalize_text(text)
    if not clean:
        return []
    return TOKEN_PATTERN.findall(clean)


def build_vocab(texts: Iterable[object], min_freq: int = 2, max_vocab: int = 50000) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(tokenize(text))

    vocab: Dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, freq in counter.most_common():
        if freq < min_freq:
            continue
        if len(vocab) >= max_vocab:
            break
        vocab[token] = len(vocab)
    return vocab


def estimate_max_length(texts: Sequence[object], lower: int = 24, upper: int = 80, percentile: float = 95.0) -> int:
    lengths = [len(tokenize(text)) for text in texts]
    if not lengths:
        return lower
    value = int(np.percentile(np.asarray(lengths, dtype=np.float32), percentile))
    return int(max(lower, min(upper, value or lower)))


def encode_texts(
    texts: Sequence[object],
    vocab: Dict[str, int],
    max_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    pad_idx = vocab[PAD_TOKEN]
    unk_idx = vocab[UNK_TOKEN]

    input_ids = np.full((len(texts), max_length), pad_idx, dtype=np.int64)
    lengths = np.zeros(len(texts), dtype=np.int64)

    for idx, text in enumerate(texts):
        tokens = tokenize(text)
        ids = [vocab.get(tok, unk_idx) for tok in tokens][:max_length]
        if not ids:
            ids = [unk_idx]
        seq_len = len(ids)
        input_ids[idx, :seq_len] = np.asarray(ids, dtype=np.int64)
        lengths[idx] = seq_len

    return input_ids, lengths


if nn is not None:

    class BiLSTMClassifier(nn.Module):
        def __init__(
            self,
            vocab_size: int,
            num_classes: int,
            embedding_dim: int = 128,
            hidden_dim: int = 128,
            num_layers: int = 1,
            dropout: float = 0.3,
            bidirectional: bool = True,
            pad_idx: int = 0,
        ) -> None:
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
            lstm_dropout = dropout if num_layers > 1 else 0.0
            self.lstm = nn.LSTM(
                embedding_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=lstm_dropout,
                bidirectional=bidirectional,
            )
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(hidden_dim * (2 if bidirectional else 1), num_classes)

        def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:  # type: ignore[name-defined]
            embeddings = self.embedding(input_ids)
            packed = pack_padded_sequence(  # type: ignore[arg-type]
                embeddings,
                lengths.detach().cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            _, (hidden, _) = self.lstm(packed)

            if self.lstm.bidirectional:
                features = torch.cat((hidden[-2], hidden[-1]), dim=1)
            else:
                features = hidden[-1]

            features = self.dropout(features)
            return self.classifier(features)

else:  # pragma: no cover - exercised only when torch is unavailable
    BiLSTMClassifier = None
