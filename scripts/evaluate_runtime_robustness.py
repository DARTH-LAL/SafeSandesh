#!/usr/bin/env python3
"""Evaluate deployed scam-detector robustness on perturbed test messages.

This script tests the same runtime stack used by the app, not only the old
baseline artifact. It runs clean messages and realistic noisy variants through
the three base models plus the deployed final ensemble, then reports macro-F1
drops for each perturbation.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import model_runtime  # noqa: E402


RISK_LABELS = ["Safe", "Suspicious", "Phishing"]
BASE_MODELS = ["baseline", "bilstm", "indicbert"]
DEFAULT_MODELS = BASE_MODELS + ["median_ensemble"]

LANGUAGE_NAMES = {
    "en": "English",
    "english": "English",
    "hi": "Hindi",
    "hindi": "Hindi",
    "pa": "Punjabi",
    "punjabi": "Punjabi",
    "ur": "Urdu",
    "urdu": "Urdu",
}

SPELLING_VARIANTS = [
    (r"\baccount\b", "acc0unt"),
    (r"\bbank\b", "b@nkp"),
    (r"\bblocked\b", "bl0cked"),
    (r"\bclick\b", "clik"),
    (r"\bconfirm\b", "cnfrm"),
    (r"\bkyc\b", "k.y.c"),
    (r"\blink\b", "l1nk"),
    (r"\blogin\b", "l0gin"),
    (r"\botp\b", "0tp"),
    (r"\bpassword\b", "passw0rd"),
    (r"\bprize\b", "pr1ze"),
    (r"\burgent\b", "urgnt"),
    (r"\bverify\b", "ver1fy"),
]

EMOJI_POOL = ["⚠️", "🔐", "🚨", "✅", "📲", "👉", "⏰", "💰"]


def parse_args() -> argparse.Namespace:
    default_test = ROOT.parent / "DATASETS" / "final_with_urdu_v3" / "splits" / "test.csv"
    default_out = ROOT / "data" / "analysis" / "runtime_robustness_v1"
    parser = argparse.ArgumentParser(
        description="Run clean/noisy robustness evaluation for the app runtime models."
    )
    parser.add_argument("--test-csv", type=Path, default=default_test)
    parser.add_argument("--out-dir", type=Path, default=default_out)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="risk_label")
    parser.add_argument("--language-column", default="language")
    parser.add_argument(
        "--models",
        default="baseline,bilstm,indicbert,median_ensemble",
        help="Comma-separated: baseline,bilstm,indicbert,median_ensemble or all.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Use 0 for the full test set. Use a smaller number for a quick smoke run.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=250)
    return parser.parse_args()


def normalize_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text == "phishing":
        return "Phishing"
    if text == "suspicious":
        return "Suspicious"
    if text == "safe":
        return "Safe"
    return "Safe"


def language_name(value: Any) -> str:
    return LANGUAGE_NAMES.get(str(value or "").strip().lower(), "English")


def normalize_model_list(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return DEFAULT_MODELS
    models = []
    for item in raw.split(","):
        model = item.strip().lower()
        if model == "final":
            model = "median_ensemble"
        if model:
            models.append(model)
    unknown = [model for model in models if model not in DEFAULT_MODELS]
    if unknown:
        raise ValueError(f"Unknown model(s): {', '.join(unknown)}")
    return models


def perturb_emoji_insertion(text: str, rng: random.Random) -> str:
    tokens = text.split()
    if not tokens:
        return text
    output: list[str] = []
    for token in tokens:
        output.append(token)
        if rng.random() < 0.14:
            output.append(rng.choice(EMOJI_POOL))
    if output == tokens:
        output.insert(min(1, len(output)), rng.choice(EMOJI_POOL))
    return " ".join(output)


def perturb_spacing_noise(text: str, rng: random.Random) -> str:
    noisy = text
    noisy = re.sub(r"([:/.-])", lambda match: f" {match.group(1)} " if rng.random() < 0.45 else match.group(1), noisy)
    noisy = re.sub(r"\s+", lambda _match: " " * rng.randint(1, 4), noisy)
    if rng.random() < 0.5:
        noisy = noisy.replace("OTP", "O T P").replace("otp", "o t p")
    return noisy.strip()


def perturb_spelling_variants(text: str, rng: random.Random) -> str:
    noisy = text
    replacements = SPELLING_VARIANTS[:]
    rng.shuffle(replacements)
    changed = False
    for pattern, replacement in replacements:
        if rng.random() < 0.65 and re.search(pattern, noisy, flags=re.IGNORECASE):
            noisy = re.sub(pattern, replacement, noisy, count=1, flags=re.IGNORECASE)
            changed = True
    if not changed and noisy:
        words = noisy.split()
        candidates = [idx for idx, word in enumerate(words) if len(word) >= 5 and word.isascii()]
        if candidates:
            idx = rng.choice(candidates)
            word = words[idx]
            cut = rng.randint(1, len(word) - 2)
            words[idx] = word[:cut] + word[cut + 1 :]
            noisy = " ".join(words)
    return noisy


PERTURBATIONS = {
    "emoji_insertion": perturb_emoji_insertion,
    "spacing_noise": perturb_spacing_noise,
    "spelling_variants": perturb_spelling_variants,
}
SCENARIO_SEED_OFFSETS = {
    "clean": 0,
    "emoji_insertion": 1000,
    "spacing_noise": 2000,
    "spelling_variants": 3000,
}


def sample_frame(df: pd.DataFrame, sample_size: int, seed: int, label_col: str, lang_col: str) -> pd.DataFrame:
    if sample_size <= 0 or sample_size >= len(df):
        return df.reset_index(drop=True).copy()

    strata_cols = [col for col in [label_col, lang_col] if col in df.columns]
    if not strata_cols:
        return df.sample(n=sample_size, random_state=seed).reset_index(drop=True)

    frac = sample_size / len(df)
    pieces: list[pd.DataFrame] = []
    for _, group in df.groupby(strata_cols, dropna=False):
        take = min(len(group), max(1, int(round(len(group) * frac))))
        pieces.append(group.sample(n=take, random_state=seed))

    sampled = pd.concat(pieces).drop_duplicates()
    if len(sampled) > sample_size:
        sampled = sampled.sample(n=sample_size, random_state=seed)
    elif len(sampled) < sample_size:
        remainder = df.drop(index=sampled.index)
        add_count = min(sample_size - len(sampled), len(remainder))
        if add_count:
            sampled = pd.concat([sampled, remainder.sample(n=add_count, random_state=seed)])
    return sampled.sample(frac=1, random_state=seed).reset_index(drop=True)


def perturb_text(text: str, scenario: str, rng: random.Random) -> str:
    if scenario == "clean":
        return text
    return PERTURBATIONS[scenario](text, rng)


def predict_stack(text: str, language: str) -> dict[str, dict[str, Any]]:
    comparison = model_runtime.compare_all_model_predictions(text, output_language=language)
    predictions: dict[str, dict[str, Any]] = {}

    for pred in comparison.get("predictions", []) or []:
        model_kind = str(pred.get("model_kind") or "").strip().lower()
        if model_kind in BASE_MODELS:
            predictions[model_kind] = {
                "label": normalize_label(pred.get("label")),
                "score": int(round(float(pred.get("risk_score") or 0))),
                "version": pred.get("model_version") or model_kind,
            }

    final = comparison.get("final") or {}
    method = comparison.get("final_score_method") or final.get("final_score_method") or "median_ensemble_v1"
    predictions["median_ensemble"] = {
        "label": normalize_label(final.get("label")),
        "score": int(round(float(final.get("risk_score") or 0))),
        "version": method,
    }
    return predictions


def compute_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    report = classification_report(
        y_true,
        y_pred,
        labels=RISK_LABELS,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=RISK_LABELS, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=RISK_LABELS, average="weighted", zero_division=0)),
        "per_label": {
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
                "support": int(report[label]["support"]),
            }
            for label in RISK_LABELS
        },
    }


def evaluate_scenario(
    df: pd.DataFrame,
    scenario: str,
    selected_models: list[str],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    y_true = [normalize_label(value) for value in df[args.label_column].tolist()]
    y_pred_by_model: dict[str, list[str]] = {model: [] for model in selected_models}
    scores_by_model: dict[str, list[int | None]] = {model: [] for model in selected_models}
    errors: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []

    for idx, row in df.iterrows():
        rng = random.Random(args.seed + (idx * 997) + SCENARIO_SEED_OFFSETS.get(scenario, 9000))
        original_text = str(row[args.text_column])
        noisy_text = perturb_text(original_text, scenario, rng)
        lang = language_name(row.get(args.language_column, "en"))

        try:
            stack = predict_stack(noisy_text, lang)
        except Exception as exc:  # noqa: BLE001
            errors.append({"row": int(idx), "scenario": scenario, "error": str(exc), "text": noisy_text[:250]})
            stack = {}

        for model in selected_models:
            pred = stack.get(model) or {}
            y_pred_by_model[model].append(normalize_label(pred.get("label")))
            score = pred.get("score")
            scores_by_model[model].append(int(score) if score is not None else None)

        if len(samples) < 12 and scenario != "clean":
            samples.append(
                {
                    "row": int(idx),
                    "language": lang,
                    "true_label": y_true[idx],
                    "scenario": scenario,
                    "original_text": original_text,
                    "perturbed_text": noisy_text,
                }
            )

        if args.progress_every and (idx + 1) % args.progress_every == 0:
            print(f"[{scenario}] scored {idx + 1}/{len(df)} rows")

    metrics_by_model: dict[str, Any] = {}
    for model in selected_models:
        metrics = compute_metrics(y_true, y_pred_by_model[model])
        metrics["mean_score"] = float(pd.Series([s for s in scores_by_model[model] if s is not None]).mean())
        metrics_by_model[model] = metrics

    return {"metrics": metrics_by_model, "errors": errors}, samples


def add_drops(results: dict[str, Any], selected_models: list[str]) -> None:
    for model in selected_models:
        clean = results["clean"]["metrics"][model]
        for scenario, payload in results.items():
            metrics = payload["metrics"][model]
            metrics["macro_f1_drop_vs_clean"] = float(clean["macro_f1"] - metrics["macro_f1"])
            metrics["accuracy_drop_vs_clean"] = float(clean["accuracy"] - metrics["accuracy"])
            metrics["phishing_f1_drop_vs_clean"] = float(
                clean["per_label"]["Phishing"]["f1"] - metrics["per_label"]["Phishing"]["f1"]
            )


def write_markdown(report: dict[str, Any], out_path: Path) -> None:
    lines = [
        "# Runtime Robustness Report",
        "",
        f"- Test CSV: `{report['test_csv']}`",
        f"- Rows evaluated: `{report['rows']}`",
        f"- Models: `{', '.join(report['models'])}`",
        f"- Perturbations: `{', '.join(report['scenarios'][1:])}`",
        "",
        "## Summary",
        "",
        "| Model | Scenario | Accuracy | Macro-F1 | Macro-F1 drop | Phishing F1 drop |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for model in report["models"]:
        for scenario in report["scenarios"]:
            metrics = report["results"][scenario]["metrics"][model]
            lines.append(
                "| "
                f"{model} | {scenario} | "
                f"{metrics['accuracy']:.4f} | "
                f"{metrics['macro_f1']:.4f} | "
                f"{metrics['macro_f1_drop_vs_clean']:.4f} | "
                f"{metrics['phishing_f1_drop_vs_clean']:.4f} |"
            )

    lines.extend(
        [
            "",
            "## How To Read This",
            "",
            "A small macro-F1 drop means the model stayed stable after realistic message noise.",
            "The report objective is met by comparing clean test-set performance against perturbed test-set performance.",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    selected_models = normalize_model_list(args.models)

    if not args.test_csv.exists():
        raise FileNotFoundError(f"Test CSV not found: {args.test_csv}")

    df = pd.read_csv(args.test_csv)
    required = [args.text_column, args.label_column]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")

    df = df[df[args.label_column].notna() & df[args.text_column].notna()].copy()
    df = sample_frame(df, args.sample_size, args.seed, args.label_column, args.language_column)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = ["clean", *PERTURBATIONS.keys()]
    results: dict[str, Any] = {}
    all_samples: list[dict[str, Any]] = []

    print(f"Evaluating {len(df)} rows across {len(scenarios)} scenarios")
    print(f"Models: {', '.join(selected_models)}")

    for scenario in scenarios:
        print(f"\nScenario: {scenario}")
        payload, samples = evaluate_scenario(df, scenario, selected_models, args)
        results[scenario] = payload
        all_samples.extend(samples)

    add_drops(results, selected_models)

    report = {
        "test_csv": str(args.test_csv),
        "rows": int(len(df)),
        "seed": int(args.seed),
        "models": selected_models,
        "scenarios": scenarios,
        "results": results,
    }

    json_path = args.out_dir / "runtime_robustness_report.json"
    md_path = args.out_dir / "runtime_robustness_report.md"
    samples_path = args.out_dir / "perturbation_samples.csv"

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, md_path)
    pd.DataFrame(all_samples).to_csv(samples_path, index=False)

    print("\nSummary")
    for model in selected_models:
        clean = results["clean"]["metrics"][model]["macro_f1"]
        drops = [
            f"{scenario}: {results[scenario]['metrics'][model]['macro_f1_drop_vs_clean']:.4f}"
            for scenario in scenarios
            if scenario != "clean"
        ]
        print(f"- {model}: clean macro-F1 {clean:.4f}; drops [{', '.join(drops)}]")

    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {samples_path}")


if __name__ == "__main__":
    main()
