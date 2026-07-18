#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASETS_ROOT = ROOT.parent / "DATASETS"

from scripts.train_baseline_model import (
    DEFAULT_SPLIT_DIR,
    DEFAULT_TYPE_CONFIDENCE_MIN,
    RISK_LABEL_ORDER,
    calibration_report_for_split,
    evaluate_type_model,
    find_best_temperature,
    label_from_score,
    load_split_csv,
    phishing_scores_from_probs,
    report_from_predictions,
    softmax,
    tier_from_score,
    train_type_pipeline,
    tune_risk_thresholds,
)
from src.bilstm import BiLSTMClassifier, build_vocab, encode_texts, estimate_max_length


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loader(
    input_ids: np.ndarray,
    lengths: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.as_tensor(input_ids, dtype=torch.long),
        torch.as_tensor(lengths, dtype=torch.long),
        torch.as_tensor(labels, dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=False)


def labels_to_indices(labels: List[str], classes: List[str]) -> np.ndarray:
    mapping = {label: idx for idx, label in enumerate(classes)}
    return np.asarray([mapping[str(label)] for label in labels], dtype=np.int64)


def class_weights_from_labels(y_idx: np.ndarray, num_classes: int) -> np.ndarray:
    counts = np.bincount(y_idx, minlength=num_classes).astype(np.float32)
    counts[counts == 0.0] = 1.0
    return (len(y_idx) / (num_classes * counts)).astype(np.float32)


def collect_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_rows: List[np.ndarray] = []
    label_rows: List[np.ndarray] = []

    with torch.inference_mode():
        for input_ids, lengths, labels in loader:
            input_ids = input_ids.to(device)
            lengths = lengths.to(device)
            logits = model(input_ids, lengths)
            logits_rows.append(logits.detach().cpu().numpy())
            label_rows.append(labels.cpu().numpy())

    if not logits_rows:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.vstack(logits_rows), np.concatenate(label_rows)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    clip_norm: float,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0

    for input_ids, lengths, labels in loader:
        input_ids = input_ids.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids, lengths)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()

        batch_size = int(labels.shape[0])
        total_loss += float(loss.item()) * batch_size
        total += batch_size
        correct += int((logits.argmax(dim=1) == labels).sum().item())

    return {
        "loss": float(total_loss / max(total, 1)),
        "accuracy": float(correct / max(total, 1)),
    }


def evaluate_risk_split(
    model: nn.Module,
    df: pd.DataFrame,
    loader: DataLoader,
    temperature: float,
    risk_classes: List[str],
    thresholds: Dict[str, float],
    vocab: Dict[str, int],
    max_length: int,
    device: torch.device,
) -> Dict:
    logits, _ = collect_logits(model, loader, device)
    y_true = df["risk_label"].astype(str).to_numpy()
    probs = softmax(logits / max(temperature, 1e-6))
    y_pred = np.array([risk_classes[i] for i in np.argmax(probs, axis=1)])
    scores = phishing_scores_from_probs(probs, risk_classes)
    y_score_pred = label_from_score(scores, thresholds)
    tiers = tier_from_score(scores, thresholds)

    out = report_from_predictions(y_true, y_pred, risk_classes)
    out["rows"] = int(len(df))
    out["temperature"] = float(temperature)
    out["score_threshold_report"] = report_from_predictions(y_true, y_score_pred, RISK_LABEL_ORDER)
    out["severity_distribution"] = {
        str(k): int(v) for k, v in pd.Series(tiers).value_counts().to_dict().items()
    }
    out["calibration"] = calibration_report_for_split(probs, y_true, risk_classes, n_bins=10)

    per_language = {}
    for language, subdf in df.groupby("language"):
        subset_texts = subdf["text"].astype(str).tolist()
        subset_labels = subdf["risk_label"].astype(str).to_numpy()
        subset_ids, subset_lengths = encode_texts(subset_texts, vocab, max_length)
        subset_loader = make_loader(
            subset_ids,
            subset_lengths,
            labels_to_indices(subset_labels.tolist(), risk_classes),
            batch_size=512,
            shuffle=False,
        )
        s_logits, _ = collect_logits(model, subset_loader, device)
        s_probs = softmax(s_logits / max(temperature, 1e-6))
        s_pred = np.array([risk_classes[i] for i in np.argmax(s_probs, axis=1)])
        s_scores = phishing_scores_from_probs(s_probs, risk_classes)
        s_score_pred = label_from_score(s_scores, thresholds)
        per_language[str(language)] = {
            **report_from_predictions(subset_labels, s_pred, risk_classes),
            "rows": int(len(subdf)),
            "score_threshold_report": report_from_predictions(subset_labels, s_score_pred, RISK_LABEL_ORDER),
        }

    out["per_language"] = per_language
    return out


def train_bilstm(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    patience: int,
    clip_norm: float,
) -> Tuple[nn.Module, Dict]:
    best_state = None
    best_epoch = 0
    best_val_macro = -1.0
    best_val_loss = float("inf")
    history: List[Dict[str, float]] = []
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, device, clip_norm)
        val_logits, val_labels = collect_logits(model, val_loader, device)
        val_loss = float(
            criterion(
                torch.as_tensor(val_logits, dtype=torch.float32, device=device),
                torch.as_tensor(val_labels, dtype=torch.long, device=device),
            ).item()
        )
        val_pred = val_logits.argmax(axis=1)
        val_macro = float(
            f1_score(
                val_labels,
                val_pred,
                labels=list(range(len(RISK_LABEL_ORDER))),
                average="macro",
                zero_division=0,
            )
        )
        val_acc = float((val_pred == val_labels).mean())

        row = {
            "epoch": float(epoch),
            "train_loss": train_stats["loss"],
            "train_accuracy": train_stats["accuracy"],
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_macro,
        }
        history.append(row)
        print(
            f"epoch {epoch:02d} | train_loss={train_stats['loss']:.4f} train_acc={train_stats['accuracy']:.4f} "
            f"| val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_macro_f1={val_macro:.4f}"
        )

        if val_macro > best_val_macro + 1e-12 or (abs(val_macro - best_val_macro) <= 1e-12 and val_loss < best_val_loss):
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            best_val_macro = float(val_macro)
            best_val_loss = float(val_loss)
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"early stopping after epoch {epoch:02d}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {
        "history": history,
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": float(best_val_macro),
        "best_val_loss": float(best_val_loss),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the BiLSTM baseline for scam risk detection.")
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=DEFAULT_SPLIT_DIR / "train.csv",
    )
    parser.add_argument(
        "--val-csv",
        type=Path,
        default=DEFAULT_SPLIT_DIR / "val.csv",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=DEFAULT_SPLIT_DIR / "test.csv",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=ROOT / "data" / "models" / "bilstm_model.joblib",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=ROOT / "data" / "models" / "bilstm_metrics.json",
    )
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--max-vocab", type=int, default=50_000)
    parser.add_argument("--max-length-upper", type=int, default=64)
    parser.add_argument("--max-length-lower", type=int, default=24)
    parser.add_argument("--length-percentile", type=float, default=95.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument(
        "--type-confidence-min",
        type=float,
        default=DEFAULT_TYPE_CONFIDENCE_MIN,
        help="Minimum classifier confidence before trusting scam-type head.",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default="bilstm_v3",
        help="Version tag embedded into model artifact and runtime output.",
    )
    args = parser.parse_args()

    seed_everything(args.seed)
    device = select_device(args.device)
    print(f"using device: {device}")

    train_df = load_split_csv(args.train_csv)
    val_df = load_split_csv(args.val_csv)
    test_df = load_split_csv(args.test_csv)

    risk_classes = sorted(train_df["risk_label"].astype(str).unique().tolist())
    train_texts = train_df["text"].astype(str).tolist()
    val_texts = val_df["text"].astype(str).tolist()
    test_texts = test_df["text"].astype(str).tolist()

    vocab = build_vocab(train_texts, min_freq=args.min_freq, max_vocab=args.max_vocab)
    max_length = estimate_max_length(
        train_texts,
        lower=args.max_length_lower,
        upper=args.max_length_upper,
        percentile=args.length_percentile,
    )
    print(f"vocab size: {len(vocab)} | max_length: {max_length}")

    train_ids, train_lengths = encode_texts(train_texts, vocab, max_length)
    val_ids, val_lengths = encode_texts(val_texts, vocab, max_length)
    test_ids, test_lengths = encode_texts(test_texts, vocab, max_length)

    train_labels = labels_to_indices(train_df["risk_label"].astype(str).tolist(), risk_classes)
    val_labels = labels_to_indices(val_df["risk_label"].astype(str).tolist(), risk_classes)
    test_labels = labels_to_indices(test_df["risk_label"].astype(str).tolist(), risk_classes)

    train_loader = make_loader(train_ids, train_lengths, train_labels, args.batch_size, shuffle=True)
    val_loader = make_loader(val_ids, val_lengths, val_labels, batch_size=512, shuffle=False)
    test_loader = make_loader(test_ids, test_lengths, test_labels, batch_size=512, shuffle=False)

    class_weights = class_weights_from_labels(train_labels, len(risk_classes))
    criterion = nn.CrossEntropyLoss(weight=torch.as_tensor(class_weights, dtype=torch.float32, device=device))

    model = BiLSTMClassifier(
        vocab_size=len(vocab),
        num_classes=len(risk_classes),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=True,
        pad_idx=vocab["<pad>"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    model, train_summary = train_bilstm(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        patience=args.patience,
        clip_norm=args.clip_norm,
    )

    val_logits, _ = collect_logits(model, val_loader, device)
    val_temperature = find_best_temperature(val_logits, val_df["risk_label"].astype(str).to_numpy(), risk_classes)
    val_probs = softmax(val_logits / max(val_temperature, 1e-6))
    val_scores = phishing_scores_from_probs(val_probs, risk_classes)
    severity_thresholds = tune_risk_thresholds(val_scores, val_df["risk_label"].astype(str).to_numpy())

    type_pipeline_non_safe, type_classes_non_safe = train_type_pipeline(train_df[train_df["risk_label"] != "Safe"].copy())
    type_pipeline_phishing, type_classes_phishing = train_type_pipeline(train_df[train_df["risk_label"] == "Phishing"].copy())

    risk_metrics = {
        "train": evaluate_risk_split(
            model,
            train_df,
            train_loader,
            val_temperature,
            risk_classes,
            severity_thresholds,
            vocab,
            max_length,
            device,
        ),
        "val": evaluate_risk_split(
            model,
            val_df,
            val_loader,
            val_temperature,
            risk_classes,
            severity_thresholds,
            vocab,
            max_length,
            device,
        ),
        "test": evaluate_risk_split(
            model,
            test_df,
            test_loader,
            val_temperature,
            risk_classes,
            severity_thresholds,
            vocab,
            max_length,
            device,
        ),
    }

    type_metrics = {
        "non_safe": {
            "train": evaluate_type_model(
                type_pipeline_non_safe,
                train_df[train_df["risk_label"] != "Safe"],
                type_classes_non_safe,
                "Type model on non-safe messages",
            ),
            "val": evaluate_type_model(
                type_pipeline_non_safe,
                val_df[val_df["risk_label"] != "Safe"],
                type_classes_non_safe,
                "Type model on non-safe messages",
            ),
            "test": evaluate_type_model(
                type_pipeline_non_safe,
                test_df[test_df["risk_label"] != "Safe"],
                type_classes_non_safe,
                "Type model on non-safe messages",
            ),
        },
        "phishing": {
            "train": evaluate_type_model(
                type_pipeline_phishing,
                train_df[train_df["risk_label"] == "Phishing"],
                type_classes_phishing,
                "Type model specialized for phishing messages",
            ),
            "val": evaluate_type_model(
                type_pipeline_phishing,
                val_df[val_df["risk_label"] == "Phishing"],
                type_classes_phishing,
                "Type model specialized for phishing messages",
            ),
            "test": evaluate_type_model(
                type_pipeline_phishing,
                test_df[test_df["risk_label"] == "Phishing"],
                type_classes_phishing,
                "Type model specialized for phishing messages",
            ),
        },
    }

    val_logits_no_temp, _ = collect_logits(model, val_loader, device)
    calibration = {
        "val": {
            "before_temperature": calibration_report_for_split(
                softmax(val_logits_no_temp),
                val_df["risk_label"].astype(str).to_numpy(),
                risk_classes,
            ),
            "after_temperature": risk_metrics["val"]["calibration"],
        },
        "test": risk_metrics["test"]["calibration"],
    }

    model_out = args.model_out.resolve()
    model_out.parent.mkdir(parents=True, exist_ok=True)
    risk_state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    bundle = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_family": "bilstm",
        "model_source": "bilstm_rnn",
        "model_version": str(args.model_version),
        "risk_model_kind": "bilstm",
        "risk_model_config": {
            "vocab_size": int(len(vocab)),
            "embedding_dim": int(args.embedding_dim),
            "hidden_dim": int(args.hidden_dim),
            "num_layers": int(args.num_layers),
            "dropout": float(args.dropout),
            "bidirectional": True,
            "pad_idx": int(vocab["<pad>"]),
        },
        "risk_model_state_dict": risk_state_dict,
        "risk_vocab": vocab,
        "risk_max_length": int(max_length),
        "risk_classes": risk_classes,
        "temperature": float(val_temperature),
        "severity_thresholds": severity_thresholds,
        "type_pipeline_non_safe": type_pipeline_non_safe,
        "type_classes_non_safe": type_classes_non_safe,
        "type_pipeline_phishing": type_pipeline_phishing,
        "type_classes_phishing": type_classes_phishing,
        "type_confidence_min": float(args.type_confidence_min),
        "label_order_hint": RISK_LABEL_ORDER,
    }
    joblib.dump(bundle, model_out)

    metrics = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_family": "bilstm",
        "model_source": "bilstm_rnn",
        "model_version": str(args.model_version),
        "device": str(device),
        "training": {
            "seed": int(args.seed),
            "epochs_requested": int(args.epochs),
            "patience": int(args.patience),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "clip_norm": float(args.clip_norm),
            "embedding_dim": int(args.embedding_dim),
            "hidden_dim": int(args.hidden_dim),
            "num_layers": int(args.num_layers),
            "dropout": float(args.dropout),
            "min_freq": int(args.min_freq),
            "max_vocab": int(args.max_vocab),
            "max_length_lower": int(args.max_length_lower),
            "max_length_upper": int(args.max_length_upper),
            "length_percentile": float(args.length_percentile),
            "vocab_size": int(len(vocab)),
            "max_length": int(max_length),
        },
        "input_splits": {
            "train_csv": str(args.train_csv.resolve()),
            "val_csv": str(args.val_csv.resolve()),
            "test_csv": str(args.test_csv.resolve()),
        },
        "rows": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "risk_classes": risk_classes,
        "temperature": float(val_temperature),
        "severity_thresholds": severity_thresholds,
        "type_confidence_min": float(args.type_confidence_min),
        "training_summary": train_summary,
        "risk_metrics": risk_metrics,
        "type_classes_non_safe": type_classes_non_safe,
        "type_classes_phishing": type_classes_phishing,
        "type_metrics": type_metrics,
        "calibration": calibration,
    }

    metrics_out = args.metrics_out.resolve()
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"saved model bundle: {model_out}")
    print(f"saved metrics: {metrics_out}")
    print(
        "val macro F1:",
        f"{risk_metrics['val']['classification_report']['macro avg']['f1-score']:.4f}",
        "| test macro F1:",
        f"{risk_metrics['test']['classification_report']['macro avg']['f1-score']:.4f}",
    )


if __name__ == "__main__":
    main()
