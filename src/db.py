import sqlite3
import os
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "app.db"
DEFAULT_SCAN_TIMEZONE = os.getenv("SCAN_TIMEZONE", "Asia/Kuala_Lumpur")


def get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def _table_columns(conn, table_name):
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {r[1] for r in rows}


def _scan_clock(ts=None):
    utc_now = datetime.now(timezone.utc)
    utc_dt = utc_now
    if ts:
        try:
            parsed = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            utc_dt = parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            utc_dt = utc_now

    try:
        local_tz = ZoneInfo(os.getenv("SCAN_TIMEZONE", DEFAULT_SCAN_TIMEZONE))
    except ZoneInfoNotFoundError:
        local_tz = datetime.now().astimezone().tzinfo or timezone.utc

    local_dt = utc_dt.astimezone(local_tz)
    return {
        "ts": utc_dt.isoformat(),
        "scan_date": local_dt.strftime("%Y-%m-%d"),
        "scan_time": local_dt.strftime("%H:%M:%S"),
        "scan_datetime_local": local_dt.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "scan_timezone": str(getattr(local_tz, "key", local_tz)),
    }


def _ensure_scan_columns(conn):
    existing = _table_columns(conn, "scans")
    wanted = {
        "scan_date": "TEXT NOT NULL DEFAULT ''",
        "scan_time": "TEXT NOT NULL DEFAULT ''",
        "scan_datetime_local": "TEXT NOT NULL DEFAULT ''",
        "scan_timezone": "TEXT NOT NULL DEFAULT ''",
        "model_confidence": "REAL NOT NULL DEFAULT 0.0",
        "model_source": "TEXT NOT NULL DEFAULT 'unknown'",
        "model_version": "TEXT NOT NULL DEFAULT 'unknown'",
        "type_source": "TEXT NOT NULL DEFAULT 'unknown'",
        "comparison_label": "TEXT NOT NULL DEFAULT 'unknown'",
        "comparison_scam_type": "TEXT NOT NULL DEFAULT 'Other'",
        "comparison_risk_score": "INTEGER NOT NULL DEFAULT 0",
        "comparison_model_confidence": "REAL NOT NULL DEFAULT 0.0",
        "comparison_model_source": "TEXT NOT NULL DEFAULT 'unknown'",
        "comparison_model_version": "TEXT NOT NULL DEFAULT 'unknown'",
        "comparison_type_source": "TEXT NOT NULL DEFAULT 'unknown'",
        "comparison_tertiary_label": "TEXT NOT NULL DEFAULT 'unknown'",
        "comparison_tertiary_scam_type": "TEXT NOT NULL DEFAULT 'Other'",
        "comparison_tertiary_risk_score": "INTEGER NOT NULL DEFAULT 0",
        "comparison_tertiary_model_confidence": "REAL NOT NULL DEFAULT 0.0",
        "comparison_tertiary_model_source": "TEXT NOT NULL DEFAULT 'unknown'",
        "comparison_tertiary_model_version": "TEXT NOT NULL DEFAULT 'unknown'",
        "comparison_tertiary_type_source": "TEXT NOT NULL DEFAULT 'unknown'",
        "review_recommended": "INTEGER NOT NULL DEFAULT 0",
        "review_reason": "TEXT NOT NULL DEFAULT ''",
        "final_score_method": "TEXT NOT NULL DEFAULT 'primary_model'",
        "model_outputs_json": "TEXT NOT NULL DEFAULT ''",
    }
    for name, ddl in wanted.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE scans ADD COLUMN {name} {ddl}")


def _backfill_scan_datetime_columns(conn):
    existing = _table_columns(conn, "scans")
    required = {"scan_date", "scan_time", "scan_datetime_local", "scan_timezone"}
    if not required.issubset(existing):
        return

    rows = conn.execute(
        """
        SELECT id, ts
        FROM scans
        WHERE scan_date = '' OR scan_time = '' OR scan_datetime_local = '' OR scan_timezone = ''
        """
    ).fetchall()
    for row_id, ts in rows:
        clock = _scan_clock(ts)
        conn.execute(
            """
            UPDATE scans
            SET scan_date = ?, scan_time = ?, scan_datetime_local = ?, scan_timezone = ?
            WHERE id = ?
            """,
            (
                clock["scan_date"],
                clock["scan_time"],
                clock["scan_datetime_local"],
                clock["scan_timezone"],
                row_id,
            ),
        )


def init_db():
    with get_conn() as conn:
        conn.execute(
            """
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            scan_date TEXT NOT NULL DEFAULT '',
            scan_time TEXT NOT NULL DEFAULT '',
            scan_datetime_local TEXT NOT NULL DEFAULT '',
            scan_timezone TEXT NOT NULL DEFAULT '',
            language TEXT NOT NULL,
            label TEXT NOT NULL,
            scam_type TEXT NOT NULL,
            risk_score INTEGER NOT NULL,
            model_confidence REAL NOT NULL DEFAULT 0.0,
            model_source TEXT NOT NULL DEFAULT 'unknown',
            model_version TEXT NOT NULL DEFAULT 'unknown',
            type_source TEXT NOT NULL DEFAULT 'unknown',
            reason TEXT NOT NULL,
            message TEXT NOT NULL,
            comparison_label TEXT NOT NULL DEFAULT 'unknown',
            comparison_scam_type TEXT NOT NULL DEFAULT 'Other',
            comparison_risk_score INTEGER NOT NULL DEFAULT 0,
            comparison_model_confidence REAL NOT NULL DEFAULT 0.0,
            comparison_model_source TEXT NOT NULL DEFAULT 'unknown',
            comparison_model_version TEXT NOT NULL DEFAULT 'unknown',
            comparison_type_source TEXT NOT NULL DEFAULT 'unknown',
            comparison_tertiary_label TEXT NOT NULL DEFAULT 'unknown',
            comparison_tertiary_scam_type TEXT NOT NULL DEFAULT 'Other',
            comparison_tertiary_risk_score INTEGER NOT NULL DEFAULT 0,
            comparison_tertiary_model_confidence REAL NOT NULL DEFAULT 0.0,
            comparison_tertiary_model_source TEXT NOT NULL DEFAULT 'unknown',
            comparison_tertiary_model_version TEXT NOT NULL DEFAULT 'unknown',
            comparison_tertiary_type_source TEXT NOT NULL DEFAULT 'unknown',
            review_recommended INTEGER NOT NULL DEFAULT 0,
            review_reason TEXT NOT NULL DEFAULT '',
            final_score_method TEXT NOT NULL DEFAULT 'primary_model',
            model_outputs_json TEXT NOT NULL DEFAULT ''
        )
        """
        )

        _ensure_scan_columns(conn)
        _backfill_scan_datetime_columns(conn)

        conn.commit()


def insert_scan(
    language,
    label,
    scam_type,
    risk_score,
    reason,
    message,
    model_confidence=0.0,
    model_source="unknown",
    model_version="unknown",
    type_source="unknown",
    comparison_label="unknown",
    comparison_scam_type="Other",
    comparison_risk_score=0,
    comparison_model_confidence=0.0,
    comparison_model_source="unknown",
    comparison_model_version="unknown",
    comparison_type_source="unknown",
    comparison_tertiary_label="unknown",
    comparison_tertiary_scam_type="Other",
    comparison_tertiary_risk_score=0,
    comparison_tertiary_model_confidence=0.0,
    comparison_tertiary_model_source="unknown",
    comparison_tertiary_model_version="unknown",
    comparison_tertiary_type_source="unknown",
    review_recommended=False,
    review_reason="",
    final_score_method="primary_model",
    model_outputs_json="",
):
    clock = _scan_clock()
    with get_conn() as conn:
        cur = conn.execute(
            """
        INSERT INTO scans (
            ts, scan_date, scan_time, scan_datetime_local, scan_timezone,
            language, label, scam_type, risk_score,
            model_confidence, model_source, model_version, type_source,
            reason, message,
            comparison_label, comparison_scam_type, comparison_risk_score,
            comparison_model_confidence, comparison_model_source,
            comparison_model_version, comparison_type_source,
            comparison_tertiary_label, comparison_tertiary_scam_type,
            comparison_tertiary_risk_score, comparison_tertiary_model_confidence,
            comparison_tertiary_model_source, comparison_tertiary_model_version,
            comparison_tertiary_type_source, review_recommended, review_reason,
            final_score_method, model_outputs_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                clock["ts"],
                clock["scan_date"],
                clock["scan_time"],
                clock["scan_datetime_local"],
                clock["scan_timezone"],
                language,
                label,
                scam_type,
                int(risk_score),
                float(model_confidence),
                str(model_source),
                str(model_version),
                str(type_source),
                reason,
                message,
                str(comparison_label),
                str(comparison_scam_type),
                int(comparison_risk_score),
                float(comparison_model_confidence),
                str(comparison_model_source),
                str(comparison_model_version),
                str(comparison_type_source),
                str(comparison_tertiary_label),
                str(comparison_tertiary_scam_type),
                int(comparison_tertiary_risk_score),
                float(comparison_tertiary_model_confidence),
                str(comparison_tertiary_model_source),
                str(comparison_tertiary_model_version),
                str(comparison_tertiary_type_source),
                int(bool(review_recommended)),
                str(review_reason or ""),
                str(final_score_method or "primary_model"),
                str(model_outputs_json or ""),
            ),
        )
        conn.commit()
        return cur.lastrowid


def read_scans(limit=5000):
    with get_conn() as conn:
        cur = conn.execute(
            """
        SELECT
            id, ts, language, label, scam_type, risk_score,
            model_confidence, model_source, model_version, type_source,
            reason, message,
            comparison_label, comparison_scam_type, comparison_risk_score,
            comparison_model_confidence, comparison_model_source,
            comparison_model_version, comparison_type_source,
            comparison_tertiary_label, comparison_tertiary_scam_type,
            comparison_tertiary_risk_score, comparison_tertiary_model_confidence,
            comparison_tertiary_model_source, comparison_tertiary_model_version,
            comparison_tertiary_type_source, review_recommended, review_reason,
            final_score_method, model_outputs_json
        FROM scans
        ORDER BY ts DESC
        LIMIT ?
        """,
            (limit,),
        )
        rows = cur.fetchall()
    return rows


def _backend_name():
    value = os.getenv("SCAN_DB_BACKEND", "")
    if not value:
        try:
            import streamlit as st

            value = st.secrets.get("SCAN_DB_BACKEND", "")
        except Exception:
            value = ""
    return str(value or "sqlite").strip().lower()


def _supabase_backend():
    from src import db_supabase

    return db_supabase


_sqlite_init_db = init_db
_sqlite_insert_scan = insert_scan
_sqlite_read_scans = read_scans


def init_db():
    if _backend_name() == "supabase":
        return _supabase_backend().init_db()
    return _sqlite_init_db()


def insert_scan(*args, **kwargs):
    if _backend_name() == "supabase":
        return _supabase_backend().insert_scan(*args, **kwargs)
    return _sqlite_insert_scan(*args, **kwargs)


def read_scans(limit=5000):
    if _backend_name() == "supabase":
        return _supabase_backend().read_scans(limit=limit)
    return _sqlite_read_scans(limit=limit)
