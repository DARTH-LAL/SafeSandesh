#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = ROOT.parent / "DATASETS"
BASE_DATASET_DIR = DATASETS_ROOT / "final_with_urdu"
DEFAULT_OUTPUT_DIR = DATASETS_ROOT / "final_with_urdu_v2"
DEFAULT_SEED_PATH = ROOT / "data" / "curation" / "multilingual_phishing_seeds_v1.json"
DEFAULT_SPLIT_SCRIPT = DATASETS_ROOT / "scripts" / "create_leakage_safe_splits.py"

RISK_RANK = {"Safe": 0, "Suspicious": 1, "Phishing": 2}
BASE_COLUMNS = [
    "record_id",
    "text",
    "text_key",
    "language",
    "risk_label",
    "risk_rank",
    "scam_type",
    "scam_type_source",
    "source_dataset",
    "source_count",
    "is_synthetic",
    "split_group",
    "preferred_split",
    "label_conflict",
    "label_options",
]


def normalize_text(text: object) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    value = unicodedata.normalize("NFKC", str(text))
    value = value.replace("\u200b", " ").replace("\ufeff", " ")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def text_key(text: str) -> str:
    return re.sub(r"\s+", " ", normalize_text(text).lower())


def hash_id(*parts: str, n: int = 16) -> str:
    blob = "||".join(parts).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:n]


def load_seed_records(path: Path) -> List[Dict[str, str]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError(f"Seed file must contain a non-empty JSON list: {path}")

    required = {
        "scenario_id",
        "curation_id",
        "language",
        "scam_type",
        "text",
        "source_title",
        "source_url",
        "source_note",
        "source_kind",
    }
    for idx, row in enumerate(records):
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"Seed row {idx} is missing required fields: {missing}")
    return records


def build_seed_frames(seed_rows: List[Dict[str, str]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dataset_rows: List[Dict[str, object]] = []
    audit_rows: List[Dict[str, object]] = []

    for row in seed_rows:
        cleaned_text = normalize_text(row["text"])
        lang = normalize_text(row["language"]).lower()
        scenario_id = normalize_text(row["scenario_id"])
        curation_id = normalize_text(row["curation_id"])
        risk_label = "Phishing"
        key = text_key(cleaned_text)
        record_id = f"msg_{hash_id(lang, key, risk_label)}"

        dataset_rows.append(
            {
                "record_id": record_id,
                "text": cleaned_text,
                "text_key": key,
                "language": lang,
                "risk_label": risk_label,
                "risk_rank": RISK_RANK[risk_label],
                "scam_type": normalize_text(row["scam_type"]) or "Other",
                "scam_type_source": "manual_source_curation",
                "source_dataset": "manual_source_curation_v1",
                "source_count": 1,
                "is_synthetic": True,
                "split_group": f"manual_{scenario_id}",
                "preferred_split": normalize_text(row.get("preferred_split")).lower(),
                "label_conflict": False,
                "label_options": risk_label,
            }
        )

        audit_rows.append(
            {
                "record_id": record_id,
                "scenario_id": scenario_id,
                "curation_id": curation_id,
                "language": lang,
                "scam_type": normalize_text(row["scam_type"]) or "Other",
                "source_title": normalize_text(row["source_title"]),
                "source_url": normalize_text(row["source_url"]),
                "source_note": normalize_text(row["source_note"]),
                "source_kind": normalize_text(row["source_kind"]) or "source_backed_paraphrase",
                "preferred_split": normalize_text(row.get("preferred_split")).lower(),
                "text": cleaned_text,
            }
        )

    dataset_df = pd.DataFrame.from_records(dataset_rows, columns=BASE_COLUMNS)
    audit_df = pd.DataFrame.from_records(audit_rows)
    return dataset_df, audit_df


def compute_non_english_phishing_counts(df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    subset = df[df["language"].isin(["hi", "pa", "ur"]) & (df["risk_label"] == "Phishing")].copy()
    return {
        "by_language": {str(k): int(v) for k, v in subset["language"].value_counts().to_dict().items()},
        "by_scam_type": {str(k): int(v) for k, v in subset["scam_type"].value_counts().to_dict().items()},
    }


def merge_rows(base_df: pd.DataFrame, curated_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    required = set(BASE_COLUMNS)
    base_work = base_df.copy()
    if "preferred_split" not in base_work.columns:
        base_work["preferred_split"] = ""
    missing = sorted(required - set(base_work.columns))
    if missing:
        raise ValueError(f"Base dataset is missing required columns: {missing}")

    base_work = base_work[BASE_COLUMNS].copy()
    curated_work = curated_df[BASE_COLUMNS].copy()

    existing_pairs = set(zip(base_work["language"].astype(str), base_work["text_key"].astype(str)))
    keep_mask = [
        (str(lang), str(key)) not in existing_pairs
        for lang, key in zip(curated_work["language"].astype(str), curated_work["text_key"].astype(str))
    ]
    added_df = curated_work.loc[keep_mask].copy()

    merged = pd.concat([base_work, added_df], ignore_index=True, sort=False)
    merged = merged.sort_values(["language", "risk_rank", "record_id"]).reset_index(drop=True)
    return merged, added_df


def write_outputs(
    merged_df: pd.DataFrame,
    added_df: pd.DataFrame,
    audit_df: pd.DataFrame,
    output_dir: Path,
    base_csv: Path,
) -> Tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "combined_messages_final.csv"
    audit_csv = output_dir / "manual_source_curation_v1.csv"
    summary_json = output_dir / "manual_source_curation_summary.json"

    merged_df.to_csv(output_csv, index=False)
    audit_df[audit_df["record_id"].isin(set(added_df["record_id"]))].copy().to_csv(audit_csv, index=False)

    summary = {
        "base_csv": str(base_csv),
        "output_csv": str(output_csv),
        "audit_csv": str(audit_csv),
        "rows_base": int(len(merged_df) - len(added_df)),
        "rows_added": int(len(added_df)),
        "rows_total": int(len(merged_df)),
        "added_by_language": {
            str(k): int(v) for k, v in added_df["language"].value_counts().to_dict().items()
        },
        "added_by_scam_type": {
            str(k): int(v) for k, v in added_df["scam_type"].value_counts().to_dict().items()
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_csv, audit_csv, summary_json


def rebuild_splits(
    python_bin: str,
    split_script: Path,
    input_csv: Path,
    output_dir: Path,
    stratify_cols: str,
) -> None:
    splits_dir = output_dir / "splits"
    cmd = [
        python_bin,
        str(split_script),
        "--input-csv",
        str(input_csv),
        "--output-dir",
        str(splits_dir),
        "--stratify-cols",
        str(stratify_cols),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch the multilingual dataset with source-backed Hindi, Punjabi, and Urdu phishing examples."
    )
    parser.add_argument(
        "--base-csv",
        type=Path,
        default=BASE_DATASET_DIR / "combined_messages_final.csv",
        help="Existing combined dataset CSV.",
    )
    parser.add_argument(
        "--seed-json",
        type=Path,
        default=DEFAULT_SEED_PATH,
        help="JSON file containing source-backed phishing examples.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the patched dataset version.",
    )
    parser.add_argument(
        "--split-script",
        type=Path,
        default=DEFAULT_SPLIT_SCRIPT,
        help="Path to the leakage-safe split builder.",
    )
    parser.add_argument(
        "--skip-splits",
        action="store_true",
        help="Write the patched dataset without rebuilding train/val/test splits.",
    )
    parser.add_argument(
        "--stratify-cols",
        type=str,
        default="language,risk_label",
        help="Comma-separated split stratification columns passed to the split builder.",
    )
    args = parser.parse_args()

    base_csv = args.base_csv.resolve()
    seed_json = args.seed_json.resolve()
    output_dir = args.output_dir.resolve()
    split_script = args.split_script.resolve()

    if not base_csv.exists():
        raise FileNotFoundError(f"Base dataset not found: {base_csv}")
    if not seed_json.exists():
        raise FileNotFoundError(f"Seed file not found: {seed_json}")
    if not split_script.exists():
        raise FileNotFoundError(f"Split script not found: {split_script}")

    base_df = pd.read_csv(base_csv)
    before_counts = compute_non_english_phishing_counts(base_df)
    seed_rows = load_seed_records(seed_json)
    curated_df, audit_df = build_seed_frames(seed_rows)
    merged_df, added_df = merge_rows(base_df, curated_df)

    output_csv, audit_csv, summary_json = write_outputs(
        merged_df=merged_df,
        added_df=added_df,
        audit_df=audit_df,
        output_dir=output_dir,
        base_csv=base_csv,
    )

    if not args.skip_splits:
        rebuild_splits(
            python_bin=sys.executable,
            split_script=split_script,
            input_csv=output_csv,
            output_dir=output_dir,
            stratify_cols=args.stratify_cols,
        )

    after_counts = compute_non_english_phishing_counts(merged_df)
    report = {
        "base_csv": str(base_csv),
        "seed_json": str(seed_json),
        "output_csv": str(output_csv),
        "audit_csv": str(audit_csv),
        "summary_json": str(summary_json),
        "rows_added": int(len(added_df)),
        "duplicates_skipped": int(len(curated_df) - len(added_df)),
        "non_english_phishing_before": before_counts,
        "non_english_phishing_after": after_counts,
        "splits_rebuilt": not args.skip_splits,
        "stratify_cols": [col.strip() for col in str(args.stratify_cols).split(",") if col.strip()],
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
