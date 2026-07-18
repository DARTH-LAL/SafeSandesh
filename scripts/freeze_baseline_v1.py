#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

try:
    import joblib
except Exception:
    joblib = None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_meta(path: Path) -> Dict:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
    }


def dir_meta(path: Path) -> Dict:
    total_bytes = 0
    total_files = 0
    for item in path.rglob("*"):
        if item.is_file():
            total_files += 1
            total_bytes += int(item.stat().st_size)
    return {
        "path": str(path.resolve()),
        "files": int(total_files),
        "bytes": int(total_bytes),
    }


def copy_and_meta(src: Path, dst: Path) -> Dict:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return file_meta(dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze a model + dataset/split hashes as a registry snapshot.")
    parser.add_argument("--release-id", type=str, default="baseline_v1")
    parser.add_argument(
        "--registry-dir",
        type=Path,
        default=Path("/Users/ajneya/Desktop/FYP/scam-webapp/data/model_registry"),
    )
    parser.add_argument(
        "--dataset-final",
        type=Path,
        default=Path("/Users/ajneya/Desktop/FYP/DATASETS/final_with_urdu/combined_messages_final.csv"),
    )
    parser.add_argument(
        "--split-train",
        type=Path,
        default=Path("/Users/ajneya/Desktop/FYP/DATASETS/final_with_urdu/splits/train.csv"),
    )
    parser.add_argument(
        "--split-val",
        type=Path,
        default=Path("/Users/ajneya/Desktop/FYP/DATASETS/final_with_urdu/splits/val.csv"),
    )
    parser.add_argument(
        "--split-test",
        type=Path,
        default=Path("/Users/ajneya/Desktop/FYP/DATASETS/final_with_urdu/splits/test.csv"),
    )
    parser.add_argument(
        "--split-summary",
        type=Path,
        default=Path("/Users/ajneya/Desktop/FYP/DATASETS/final_with_urdu/splits/split_summary.json"),
    )
    parser.add_argument(
        "--model-file",
        type=Path,
        default=Path("/Users/ajneya/Desktop/FYP/scam-webapp/data/models/baseline_model.joblib"),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Optional directory with extra model files to copy into the registry snapshot.",
    )
    parser.add_argument(
        "--metrics-file",
        type=Path,
        default=Path("/Users/ajneya/Desktop/FYP/scam-webapp/data/models/baseline_metrics.json"),
    )
    parser.add_argument(
        "--trainer-script",
        type=Path,
        default=Path("/Users/ajneya/Desktop/FYP/scam-webapp/scripts/train_baseline_model.py"),
    )
    args = parser.parse_args()

    required = [
        args.dataset_final,
        args.split_train,
        args.split_val,
        args.split_test,
        args.split_summary,
        args.model_file,
        args.metrics_file,
        args.trainer_script,
    ]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(f"Required file missing: {p}")

    release_dir = (args.registry_dir / args.release_id).resolve()
    artifacts_dir = release_dir / "artifacts"
    release_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    copied = {
        "model": copy_and_meta(args.model_file, artifacts_dir / args.model_file.name),
        "metrics": copy_and_meta(args.metrics_file, artifacts_dir / args.metrics_file.name),
        "split_summary": copy_and_meta(args.split_summary, artifacts_dir / "split_summary.json"),
    }

    extra_model_dir = args.model_dir
    if extra_model_dir is None and joblib is not None:
        try:
            bundle = joblib.load(args.model_file)
            raw_dir = bundle.get("risk_model_dir")
            if raw_dir:
                extra_model_dir = Path(str(raw_dir))
        except Exception:
            extra_model_dir = None

    if extra_model_dir is not None:
        if not extra_model_dir.is_absolute():
            extra_model_dir = args.model_file.parent / extra_model_dir
        if extra_model_dir.exists():
            dst_dir = artifacts_dir / extra_model_dir.name
            if dst_dir.exists():
                shutil.rmtree(dst_dir)
            shutil.copytree(extra_model_dir, dst_dir)
            copied["model_dir"] = dir_meta(dst_dir)

    manifest = {
        "release_id": args.release_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "notes": "Frozen model snapshot for reproducible future comparisons.",
        "dataset": {
            "final_dataset": file_meta(args.dataset_final),
        },
        "splits": {
            "train": file_meta(args.split_train),
            "val": file_meta(args.split_val),
            "test": file_meta(args.split_test),
            "summary": file_meta(args.split_summary),
        },
        "model_training": {
            "trainer_script": file_meta(args.trainer_script),
            "model_artifact": file_meta(args.model_file),
            "metrics_artifact": file_meta(args.metrics_file),
        },
        "frozen_artifacts": copied,
    }

    manifest_path = release_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    readme_path = release_dir / "README.md"
    readme = f"""# {args.release_id}

Frozen model registry snapshot.

- Manifest: `{manifest_path}`
- Artifacts directory: `{artifacts_dir}`

Use this release ID for all future benchmark comparisons.
"""
    readme_path.write_text(readme, encoding="utf-8")

    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote readme: {readme_path}")
    print(f"Frozen artifacts: {artifacts_dir}")


if __name__ == "__main__":
    main()
