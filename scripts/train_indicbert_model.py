#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_transformer_model import main as train_transformer_main


DEFAULT_ARGS = [
    "--model-name",
    "ai4bharat/IndicBERTv2-MLM-only",
    "--model-kind",
    "indicbert",
    "--model-source",
    "indicbert_multilingual",
    "--model-version",
    "indicbert_v3",
    "--model-dir",
    str(ROOT / "data" / "models" / "indicbert_model"),
    "--model-out",
    str(ROOT / "data" / "models" / "indicbert_model.joblib"),
    "--metrics-out",
    str(ROOT / "data" / "models" / "indicbert_metrics.json"),
    "--epochs",
    "3",
    "--patience",
    "1",
    "--batch-size",
    "16",
    "--max-length-lower",
    "32",
    "--max-length-upper",
    "64",
    "--length-percentile",
    "95",
]


if __name__ == "__main__":
    sys.argv = [sys.argv[0], *DEFAULT_ARGS, *sys.argv[1:]]
    train_transformer_main()
