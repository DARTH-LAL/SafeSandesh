#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASETS_ROOT = ROOT.parent / "DATASETS"

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
from src.transformer_model import DEFAULT_TRANSFORMER_NAME


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


class TextLabelDataset(Dataset):
    def __init__(self, texts: List[str], labels: np.ndarray) -> None:
        self.texts = list(texts)
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[str, torch.Tensor]:
        return self.texts[idx], self.labels[idx]


def make_loader(
    texts: List[str],
    labels: np.ndarray,
    tokenizer,
    max_length: int,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TextLabelDataset(texts, labels)

    def collate(batch):
        batch_texts, batch_labels = zip(*batch)
        enc = tokenizer(
            list(batch_texts),
            truncation=True,
            padding=True,
            max_length=int(max_length),
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors="pt",
        )
        return enc["input_ids"], enc["attention_mask"], torch.stack(list(batch_labels))

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=False, collate_fn=collate)


def labels_to_indices(labels: List[str], classes: List[str]) -> np.ndarray:
    mapping = {label: idx for idx, label in enumerate(classes)}
    return np.asarray([mapping[str(label)] for label in labels], dtype=np.int64)


def estimate_transformer_max_length(
    texts: List[str],
    tokenizer,
    lower: int = 32,
    upper: int = 64,
    percentile: float = 95.0,
) -> int:
    if not texts:
        return int(lower)

    lengths = []
    for text in texts:
        encoded = tokenizer(
            str(text),
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        lengths.append(len(encoded["input_ids"]))

    value = int(np.percentile(np.asarray(lengths, dtype=np.float32), percentile))
    return int(max(lower, min(upper, value or lower)))


def class_weights_from_labels(y_idx: np.ndarray, num_classes: int) -> np.ndarray:
    counts = np.bincount(y_idx, minlength=num_classes).astype(np.float32)
    counts[counts == 0.0] = 1.0
    return (len(y_idx) / (num_classes * counts)).astype(np.float32)


def _get_transformer_base_model(model: nn.Module) -> nn.Module | None:
    base_prefix = getattr(model, "base_model_prefix", None)
    return getattr(model, str(base_prefix), None) if base_prefix else None


def _get_transformer_layers(base_model: nn.Module) -> List[nn.Module]:
    for holder_name in ("transformer", "encoder"):
        holder = getattr(base_model, holder_name, None)
        if holder is None:
            continue
        layers = getattr(holder, "layer", None)
        if layers is not None:
            return list(layers)

    layers = getattr(base_model, "layer", None)
    if layers is not None:
        return list(layers)
    return []


def configure_transformer_trainable_params(model: nn.Module, unfreeze_last_layers: int) -> Dict[str, int]:
    base_model = _get_transformer_base_model(model)
    if base_model is None:
        raise RuntimeError("Could not locate the transformer base model.")

    for param in base_model.parameters():
        param.requires_grad = False

    head_params = 0
    for head_name in ("pre_classifier", "classifier", "score"):
        head = getattr(model, head_name, None)
        if head is not None:
            for param in head.parameters():
                param.requires_grad = True
                head_params += int(param.numel())

    backbone_params = 0
    layers = _get_transformer_layers(base_model)
    if unfreeze_last_layers > 0 and layers:
        for layer in layers[-int(unfreeze_last_layers) :]:
            for param in layer.parameters():
                param.requires_grad = True
                backbone_params += int(param.numel())

        for norm_name in ("layer_norm", "final_layer_norm"):
            norm = getattr(base_model, norm_name, None)
            if norm is not None:
                for param in norm.parameters():
                    param.requires_grad = True
                    backbone_params += int(param.numel())

    return {"backbone_params": backbone_params, "head_params": head_params}


def build_scheduler(optimizer, total_steps: int, warmup_steps: int) -> LambdaLR:
    warmup_steps = max(0, int(warmup_steps))
    total_steps = max(1, int(total_steps))

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        remaining = total_steps - max(current_step, warmup_steps)
        decay_steps = max(1, total_steps - warmup_steps)
        return max(0.0, float(remaining) / float(decay_steps))

    return LambdaLR(optimizer, lr_lambda)


def collect_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_rows: List[np.ndarray] = []
    label_rows: List[np.ndarray] = []

    with torch.inference_mode():
        for input_ids, attention_mask, labels in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
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
    scheduler: LambdaLR,
    device: torch.device,
    clip_norm: float,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0
    report_every = max(1, len(loader) // 5)

    for step, (input_ids, attention_mask, labels) in enumerate(loader, start=1):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        scheduler.step()

        batch_size = int(labels.shape[0])
        total_loss += float(loss.item()) * batch_size
        total += batch_size
        correct += int((logits.argmax(dim=1) == labels).sum().item())

        if step % report_every == 0 or step == len(loader):
            print(f"  batch {step:>4}/{len(loader)} loss={float(loss.item()):.4f}", flush=True)

    return {
        "loss": float(total_loss / max(total, 1)),
        "accuracy": float(correct / max(total, 1)),
    }


def evaluate_risk_split(
    model: nn.Module,
    tokenizer,
    df: pd.DataFrame,
    texts: List[str],
    labels: np.ndarray,
    temperature: float,
    risk_classes: List[str],
    thresholds: Dict[str, float],
    max_length: int,
    device: torch.device,
) -> Dict:
    loader = make_loader(texts, labels, tokenizer, max_length, batch_size=64, shuffle=False)
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
        sx = subdf["text"].astype(str).tolist()
        sy_true = subdf["risk_label"].astype(str).to_numpy()
        s_labels = labels_to_indices(sy_true.tolist(), risk_classes)
        s_loader = make_loader(sx, s_labels, tokenizer, max_length, batch_size=64, shuffle=False)
        s_logits, _ = collect_logits(model, s_loader, device)
        s_probs = softmax(s_logits / max(temperature, 1e-6))
        s_pred = np.array([risk_classes[i] for i in np.argmax(s_probs, axis=1)])
        s_scores = phishing_scores_from_probs(s_probs, risk_classes)
        s_score_pred = label_from_score(s_scores, thresholds)
        per_language[str(language)] = {
            **report_from_predictions(sy_true, s_pred, risk_classes),
            "rows": int(len(subdf)),
            "score_threshold_report": report_from_predictions(sy_true, s_score_pred, RISK_LABEL_ORDER),
        }

    out["per_language"] = per_language
    return out


def train_transformer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
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
        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, clip_norm)
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
    parser = argparse.ArgumentParser(description="Fine-tune a multilingual transformer for scam risk detection.")
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
        "--model-name",
        type=str,
        default=DEFAULT_TRANSFORMER_NAME,
        help="Hugging Face model name to fine-tune.",
    )
    parser.add_argument(
        "--model-kind",
        type=str,
        default="indicbert",
        help="Runtime model kind stored in the saved bundle.",
    )
    parser.add_argument(
        "--model-source",
        type=str,
        default="indicbert_multilingual",
        help="Human-readable model source stored in the saved bundle.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=ROOT / "data" / "models" / "indicbert_model",
        help="Directory where the fine-tuned transformer weights will be saved.",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=ROOT / "data" / "models" / "indicbert_model.joblib",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=ROOT / "data" / "models" / "indicbert_metrics.json",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument(
        "--backbone-learning-rate",
        type=float,
        default=5e-6,
        help="Learning rate for unfrozen transformer backbone layers.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--max-length-upper", type=int, default=64)
    parser.add_argument("--max-length-lower", type=int, default=32)
    parser.add_argument("--length-percentile", type=float, default=95.0)
    parser.add_argument(
        "--unfreeze-last-layers",
        type=int,
        default=4,
        help="Number of final transformer encoder layers to unfreeze.",
    )
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
        default="indicbert_v3",
        help="Version tag embedded into model artifact and runtime output.",
    )
    args = parser.parse_args()

    seed_everything(args.seed)
    device = select_device(args.device)
    print(f"using device: {device}")

    train_df = load_split_csv(args.train_csv)
    val_df = load_split_csv(args.val_csv)
    test_df = load_split_csv(args.test_csv)

    risk_classes = list(RISK_LABEL_ORDER)
    train_texts = train_df["text"].astype(str).tolist()
    val_texts = val_df["text"].astype(str).tolist()
    test_texts = test_df["text"].astype(str).tolist()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    max_length = estimate_transformer_max_length(
        train_texts,
        tokenizer,
        lower=args.max_length_lower,
        upper=args.max_length_upper,
        percentile=args.length_percentile,
    )
    print(
        f"model: {args.model_name} | max_length: {max_length} "
        f"(percentile={args.length_percentile:.1f}, bounds={args.max_length_lower}-{args.max_length_upper})"
    )

    train_labels = labels_to_indices(train_df["risk_label"].astype(str).tolist(), risk_classes)
    val_labels = labels_to_indices(val_df["risk_label"].astype(str).tolist(), risk_classes)
    test_labels = labels_to_indices(test_df["risk_label"].astype(str).tolist(), risk_classes)

    train_loader = make_loader(train_texts, train_labels, tokenizer, max_length, args.batch_size, shuffle=True)
    val_loader = make_loader(val_texts, val_labels, tokenizer, max_length, batch_size=32, shuffle=False)
    test_loader = make_loader(test_texts, test_labels, tokenizer, max_length, batch_size=32, shuffle=False)

    class_weights = class_weights_from_labels(train_labels, len(risk_classes))
    criterion = nn.CrossEntropyLoss(weight=torch.as_tensor(class_weights, dtype=torch.float32, device=device))

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=len(risk_classes))
    model.config.id2label = {idx: label for idx, label in enumerate(risk_classes)}
    model.config.label2id = {label: idx for idx, label in enumerate(risk_classes)}
    model.config.problem_type = "single_label_classification"
    trainable_counts = configure_transformer_trainable_params(model, args.unfreeze_last_layers)
    model.to(device)

    head_param_ids = set()
    for head_name in ("pre_classifier", "classifier", "score"):
        head = getattr(model, head_name, None)
        if head is not None:
            for param in head.parameters():
                if param.requires_grad:
                    head_param_ids.add(id(param))

    backbone_parameters = [
        p for p in model.parameters() if p.requires_grad and id(p) not in head_param_ids
    ]
    head_parameters = [p for p in model.parameters() if p.requires_grad and id(p) in head_param_ids]

    optimizer_groups = []
    if backbone_parameters:
        optimizer_groups.append(
            {
                "params": backbone_parameters,
                "lr": args.backbone_learning_rate,
                "weight_decay": args.weight_decay,
            }
        )
    if head_parameters:
        optimizer_groups.append(
            {
                "params": head_parameters,
                "lr": args.learning_rate,
                "weight_decay": args.weight_decay,
            }
        )
    if not optimizer_groups:
        raise RuntimeError("No trainable parameters found for transformer fine-tuning.")

    optimizer = torch.optim.AdamW(optimizer_groups)
    total_steps = max(1, len(train_loader) * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = build_scheduler(optimizer, total_steps=total_steps, warmup_steps=warmup_steps)

    model, train_summary = train_transformer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
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
            tokenizer,
            train_df,
            train_texts,
            train_labels,
            val_temperature,
            risk_classes,
            severity_thresholds,
            max_length,
            device,
        ),
        "val": evaluate_risk_split(
            model,
            tokenizer,
            val_df,
            val_texts,
            val_labels,
            val_temperature,
            risk_classes,
            severity_thresholds,
            max_length,
            device,
        ),
        "test": evaluate_risk_split(
            model,
            tokenizer,
            test_df,
            test_texts,
            test_labels,
            val_temperature,
            risk_classes,
            severity_thresholds,
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

    model_dir = args.model_dir.resolve()
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    bundle = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_family": "transformer",
        "model_source": str(args.model_source),
        "model_version": str(args.model_version),
        "risk_model_kind": str(args.model_kind),
        "risk_model_name": str(args.model_name),
        "risk_model_dir": model_dir.name,
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

    model_out = args.model_out.resolve()
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_out)

    metrics = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_family": "transformer",
        "model_source": str(args.model_source),
        "model_version": str(args.model_version),
        "risk_model_name": str(args.model_name),
        "risk_model_dir": str(model_dir),
        "device": str(device),
        "training": {
            "seed": int(args.seed),
            "epochs_requested": int(args.epochs),
            "patience": int(args.patience),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "backbone_learning_rate": float(args.backbone_learning_rate),
            "weight_decay": float(args.weight_decay),
            "warmup_ratio": float(args.warmup_ratio),
            "clip_norm": float(args.clip_norm),
            "max_length_lower": int(args.max_length_lower),
            "max_length_upper": int(args.max_length_upper),
            "length_percentile": float(args.length_percentile),
            "max_length": int(max_length),
            "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "trainable_backbone_parameters": int(trainable_counts["backbone_params"]),
            "trainable_head_parameters": int(trainable_counts["head_params"]),
            "unfreeze_last_layers": int(args.unfreeze_last_layers),
            "total_parameters": int(sum(p.numel() for p in model.parameters())),
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
    print(f"saved model dir: {model_dir}")
    print(f"saved metrics: {metrics_out}")
    print(
        "val macro F1:",
        f"{risk_metrics['val']['classification_report']['macro avg']['f1-score']:.4f}",
        "| test macro F1:",
        f"{risk_metrics['test']['classification_report']['macro avg']['f1-score']:.4f}",
    )


if __name__ == "__main__":
    main()
