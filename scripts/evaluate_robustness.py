#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Callable, Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report


EMOJIS = ["⚠️", "🔒", "💳", "📲", "🛑", "✅", "💰", "📦", "🎁", "⏳"]


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / exp_vals.sum(axis=1, keepdims=True)


def perturb_emoji_insertion(text: str, rng: random.Random) -> str:
    tokens = text.split()
    if len(tokens) <= 2:
        return text
    out = []
    for tok in tokens:
        out.append(tok)
        if rng.random() < 0.18:
            out.append(rng.choice(EMOJIS))
    return " ".join(out)


def perturb_spacing_noise(text: str, rng: random.Random) -> str:
    t = re.sub(
        r"([,.!?;:])",
        lambda m: (" " if rng.random() < 0.5 else "") + m.group(1) + (" " if rng.random() < 0.7 else ""),
        text,
    )
    t = re.sub(r"\s+", " ", t).strip()
    words = t.split()
    if len(words) > 3:
        for _ in range(min(2, len(words) // 8 + 1)):
            i = rng.randrange(1, len(words))
            words[i] = (" " + words[i]) if rng.random() < 0.5 else (words[i] + " ")
    return " ".join(words).strip()


def _mutate_word(word: str, rng: random.Random) -> str:
    if len(word) < 5:
        return word

    if rng.random() < 0.5:
        for v in "aeiouAEIOU":
            idx = word.find(v)
            if idx > 0:
                return word[:idx] + word[idx + 1 :]
        return word

    i = rng.randrange(1, len(word) - 1)
    chars = list(word)
    chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


def perturb_spelling_variants(text: str, rng: random.Random) -> str:
    tokens = text.split()
    if len(tokens) <= 2:
        return text
    out = []
    for tok in tokens:
        clean = re.sub(r"[^\w]", "", tok)
        if len(clean) >= 5 and rng.random() < 0.22:
            out.append(_mutate_word(tok, rng))
        else:
            out.append(tok)
    return " ".join(out)


def evaluate_split(
    df: pd.DataFrame,
    texts: List[str],
    risk_pipeline,
    risk_classes: List[str],
    temperature: float,
) -> Dict:
    logits = risk_pipeline.decision_function(texts)
    probs = softmax(logits / temperature)
    pred_idx = np.argmax(probs, axis=1)
    pred_labels = np.array([risk_classes[i] for i in pred_idx])

    report = classification_report(
        df["risk_label"].astype(str).to_numpy(),
        pred_labels,
        labels=["Safe", "Suspicious", "Phishing"],
        output_dict=True,
        zero_division=0,
    )
    return {
        "rows": int(len(df)),
        "accuracy": float(report["accuracy"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "safe_f1": float(report["Safe"]["f1-score"]),
        "suspicious_f1": float(report["Suspicious"]["f1-score"]),
        "phishing_f1": float(report["Phishing"]["f1-score"]),
        "classification_report": report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Robustness evaluation with realistic perturbations.")
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=Path("/Users/ajneya/Desktop/FYP/DATASETS/final_with_urdu/splits/test.csv"),
    )
    parser.add_argument(
        "--model-file",
        type=Path,
        default=Path("/Users/ajneya/Desktop/FYP/scam-webapp/data/models/baseline_model.joblib"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/Users/ajneya/Desktop/FYP/scam-webapp/data/analysis/robustness_v1"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    test_df = pd.read_csv(args.test_csv)
    bundle = joblib.load(args.model_file)
    risk_pipeline = bundle["risk_pipeline"]
    risk_classes = bundle["risk_classes"]
    temperature = float(bundle.get("temperature", 1.0))

    base_texts = test_df["text"].astype(str).tolist()
    clean = evaluate_split(test_df, base_texts, risk_pipeline, risk_classes, temperature)

    perturbations: Dict[str, Callable[[str, random.Random], str]] = {
        "emoji_insertion": perturb_emoji_insertion,
        "spacing_noise": perturb_spacing_noise,
        "spelling_variants": perturb_spelling_variants,
    }

    results = {"clean": clean, "perturbations": {}, "seed": args.seed}

    sample_rows = []
    for name, fn in perturbations.items():
        pert_texts = [fn(text, rng) for text in base_texts]
        metrics = evaluate_split(test_df, pert_texts, risk_pipeline, risk_classes, temperature)
        metrics["delta_vs_clean"] = {
            "accuracy_drop": float(clean["accuracy"] - metrics["accuracy"]),
            "macro_f1_drop": float(clean["macro_f1"] - metrics["macro_f1"]),
            "weighted_f1_drop": float(clean["weighted_f1"] - metrics["weighted_f1"]),
            "phishing_f1_drop": float(clean["phishing_f1"] - metrics["phishing_f1"]),
            "suspicious_f1_drop": float(clean["suspicious_f1"] - metrics["suspicious_f1"]),
        }
        results["perturbations"][name] = metrics

        for i in [5, 55, 555]:
            if i < len(base_texts):
                sample_rows.append(
                    {
                        "perturbation": name,
                        "index": i,
                        "original": base_texts[i],
                        "perturbed": pert_texts[i],
                    }
                )

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "robustness_report.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    samples_path = out_dir / "perturbation_samples.csv"
    pd.DataFrame(sample_rows).to_csv(samples_path, index=False)

    lines = []
    for name, data in results["perturbations"].items():
        d = data["delta_vs_clean"]
        lines.append(
            f"- `{name}`: macro-F1 drop={d['macro_f1_drop']:.4f}, accuracy drop={d['accuracy_drop']:.4f}, phishing F1 drop={d['phishing_f1_drop']:.4f}"
        )

    md = f"""# Robustness Evaluation (v1)

Clean test macro-F1: **{clean['macro_f1']:.4f}**

## Performance drops vs clean
{chr(10).join(lines)}

## Files
- `{json_path}`
- `{samples_path}`
"""
    md_path = out_dir / "robustness_report.md"
    md_path.write_text(md, encoding="utf-8")

    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")
    print(f"Wrote: {samples_path}")


if __name__ == "__main__":
    main()
