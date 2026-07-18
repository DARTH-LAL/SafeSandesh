from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import db as sqlite_db  # noqa: E402
from src import db_supabase  # noqa: E402


def _quote_identifier(value: str) -> str:
    if not value.replace("_", "").isalnum():
        raise ValueError(f"Unsafe SQLite identifier: {value}")
    return f'"{value}"'


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    return {row["name"] for row in conn.execute(f"PRAGMA table_info({_quote_identifier(table_name)})")}


def _read_table(conn: sqlite3.Connection, table_name: str, columns: list[str]) -> list[dict[str, Any]]:
    if not _table_exists(conn, table_name):
        return []

    existing = _table_columns(conn, table_name)
    selected = [column for column in columns if column in existing]
    if not selected:
        return []

    column_sql = ", ".join(_quote_identifier(column) for column in selected)
    order_sql = " order by id asc" if "id" in existing else ""
    rows = conn.execute(f"select {column_sql} from {_quote_identifier(table_name)}{order_sql}").fetchall()
    return [{column: row[column] for column in selected} for row in rows]


def _load_sqlite_records(sqlite_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not sqlite_path.exists():
        raise FileNotFoundError(f"Local SQLite database not found: {sqlite_path}")

    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    try:
        scans = _read_table(conn, "scans", db_supabase.SCAN_STORAGE_COLUMNS)
        feedback = _read_table(conn, "feedback", db_supabase.FEEDBACK_COLUMNS)
    finally:
        conn.close()
    return scans, feedback


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Copy local SQLite scan history and feedback into Supabase."
    )
    parser.add_argument(
        "--sqlite-path",
        type=Path,
        default=sqlite_db.DB_PATH,
        help="Path to local app.db. Defaults to data/app.db.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Supabase upsert batch size.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count local records without connecting to Supabase.",
    )
    parser.add_argument(
        "--skip-sequence-reset",
        action="store_true",
        help="Skip resetting Supabase identity counters after preserving SQLite IDs.",
    )
    args = parser.parse_args()

    scans, feedback = _load_sqlite_records(args.sqlite_path)
    print(f"SQLite source: {args.sqlite_path}")
    print(f"Found {len(scans)} scan rows and {len(feedback)} feedback rows.")

    if args.dry_run:
        print("Dry run only. No Supabase writes were made.")
        return 0

    if scans:
        migrated = db_supabase.upsert_scan_records(scans, batch_size=args.batch_size)
        print(f"Migrated {migrated} scan rows to Supabase.")
    else:
        print("No scan rows to migrate.")

    if feedback:
        migrated = db_supabase.upsert_feedback_records(feedback, batch_size=args.batch_size)
        print(f"Migrated {migrated} feedback rows to Supabase.")
    else:
        print("No feedback rows to migrate.")

    if not args.skip_sequence_reset:
        try:
            db_supabase.reset_identity_sequences()
            print("Reset Supabase identity counters after migration.")
        except Exception as exc:
            print(f"Warning: migration finished, but identity reset failed: {exc}")
            print("If new inserts fail later, rerun scripts/supabase_schema.sql and migrate again.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
