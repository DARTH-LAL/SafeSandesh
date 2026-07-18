from __future__ import annotations

import json
import os
import ssl
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

try:
    import certifi
except Exception:
    certifi = None

DEFAULT_SCAN_TIMEZONE = os.getenv("SCAN_TIMEZONE", "Asia/Kuala_Lumpur")


SCAN_COLUMNS = [
    "id",
    "ts",
    "language",
    "label",
    "scam_type",
    "risk_score",
    "model_confidence",
    "model_source",
    "model_version",
    "type_source",
    "reason",
    "message",
    "comparison_label",
    "comparison_scam_type",
    "comparison_risk_score",
    "comparison_model_confidence",
    "comparison_model_source",
    "comparison_model_version",
    "comparison_type_source",
    "comparison_tertiary_label",
    "comparison_tertiary_scam_type",
    "comparison_tertiary_risk_score",
    "comparison_tertiary_model_confidence",
    "comparison_tertiary_model_source",
    "comparison_tertiary_model_version",
    "comparison_tertiary_type_source",
    "review_recommended",
    "review_reason",
    "final_score_method",
    "model_outputs_json",
]

SCAN_DATETIME_COLUMNS = [
    "scan_date",
    "scan_time",
    "scan_datetime_local",
    "scan_timezone",
]

SCAN_STORAGE_COLUMNS = [
    "id",
    "ts",
    *SCAN_DATETIME_COLUMNS,
    *SCAN_COLUMNS[2:],
]

FEEDBACK_COLUMNS = [
    "id",
    "ts",
    "language",
    "pred_label",
    "pred_scam_type",
    "pred_risk_score",
    "pred_confidence",
    "model_source",
    "model_version",
    "feedback_type",
    "expected_label",
    "expected_scam_type",
    "feedback_note",
    "message",
]


def _secret_value(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value:
        return value.strip()

    try:
        import streamlit as st

        value = st.secrets.get(name, default)
        if value:
            return str(value).strip()
    except Exception:
        pass

    return default


def _config() -> tuple[str, str]:
    url = _secret_value("SUPABASE_URL").rstrip("/")
    key = (
        _secret_value("SUPABASE_SERVICE_ROLE_KEY")
        or _secret_value("SUPABASE_KEY")
        or _secret_value("SUPABASE_ANON_KEY")
    )
    if not url or not key:
        raise RuntimeError(
            "Supabase backend selected, but SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY "
            "or SUPABASE_ANON_KEY are not configured."
        )
    return url, key


@lru_cache(maxsize=1)
def _ssl_context() -> ssl.SSLContext:
    if certifi is not None:
        return ssl.create_default_context(cafile=certifi.where())
    return ssl.create_default_context()


def _request(
    method: str,
    path: str,
    payload: Any | None = None,
    *,
    prefer: str = "return=representation",
) -> Any:
    url, key = _config()
    endpoint = f"{url}/rest/v1/{path.lstrip('/')}"
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    if prefer:
        headers["Prefer"] = prefer

    body = None
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    request = urllib.request.Request(endpoint, data=body, headers=headers, method=method.upper())
    try:
        with urllib.request.urlopen(request, timeout=30, context=_ssl_context()) as response:
            data = response.read().decode("utf-8")
            return json.loads(data) if data else None
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Supabase request failed: {exc.code} {exc.reason}. {details}") from exc


def _select(table: str, columns: list[str], limit: int = 5000) -> list[dict[str, Any]]:
    query = urllib.parse.urlencode(
        {
            "select": ",".join(columns),
            "order": "ts.desc",
            "limit": str(int(limit)),
        }
    )
    rows = _request("GET", f"{table}?{query}", prefer="")
    return rows if isinstance(rows, list) else []


def _row_tuple(row: dict[str, Any], columns: list[str]) -> tuple:
    return tuple(row.get(column) for column in columns)


def _scan_clock(ts: Any | None = None) -> dict[str, str]:
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


def _clean_scan_record(record: dict[str, Any]) -> dict[str, Any]:
    out = {column: record.get(column) for column in SCAN_STORAGE_COLUMNS if column in record}
    clock = _scan_clock(out.get("ts"))
    out.setdefault("ts", clock["ts"])
    out.setdefault("scan_date", clock["scan_date"])
    out.setdefault("scan_time", clock["scan_time"])
    out.setdefault("scan_datetime_local", clock["scan_datetime_local"])
    out.setdefault("scan_timezone", clock["scan_timezone"])
    out.setdefault("language", "English")
    out.setdefault("label", "unknown")
    out.setdefault("scam_type", "Other")
    out["risk_score"] = int(out.get("risk_score") or 0)
    out["model_confidence"] = float(out.get("model_confidence") or 0.0)
    out.setdefault("model_source", "unknown")
    out.setdefault("model_version", "unknown")
    out.setdefault("type_source", "unknown")
    out.setdefault("reason", "")
    out.setdefault("message", "")
    out.setdefault("comparison_label", "unknown")
    out.setdefault("comparison_scam_type", "Other")
    out["comparison_risk_score"] = int(out.get("comparison_risk_score") or 0)
    out["comparison_model_confidence"] = float(out.get("comparison_model_confidence") or 0.0)
    out.setdefault("comparison_model_source", "unknown")
    out.setdefault("comparison_model_version", "unknown")
    out.setdefault("comparison_type_source", "unknown")
    out.setdefault("comparison_tertiary_label", "unknown")
    out.setdefault("comparison_tertiary_scam_type", "Other")
    out["comparison_tertiary_risk_score"] = int(out.get("comparison_tertiary_risk_score") or 0)
    out["comparison_tertiary_model_confidence"] = float(out.get("comparison_tertiary_model_confidence") or 0.0)
    out.setdefault("comparison_tertiary_model_source", "unknown")
    out.setdefault("comparison_tertiary_model_version", "unknown")
    out.setdefault("comparison_tertiary_type_source", "unknown")
    out["review_recommended"] = bool(out.get("review_recommended") or False)
    out.setdefault("review_reason", "")
    out.setdefault("final_score_method", "primary_model")
    out.setdefault("model_outputs_json", "")
    return out


def _clean_feedback_record(record: dict[str, Any]) -> dict[str, Any]:
    out = {column: record.get(column) for column in FEEDBACK_COLUMNS if column in record}
    out.setdefault("ts", datetime.utcnow().isoformat())
    out.setdefault("language", "English")
    out.setdefault("message", "")
    out.setdefault("pred_label", "unknown")
    out.setdefault("pred_scam_type", "Other")
    out["pred_risk_score"] = int(out.get("pred_risk_score") or 0)
    out["pred_confidence"] = float(out.get("pred_confidence") or 0.0)
    out.setdefault("model_source", "unknown")
    out.setdefault("model_version", "unknown")
    out.setdefault("feedback_type", "wrong_prediction")
    return out


def init_db() -> None:
    _config()


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
    record = _clean_scan_record(
        {
            "language": language,
            "label": label,
            "scam_type": scam_type,
            "risk_score": risk_score,
            "model_confidence": model_confidence,
            "model_source": model_source,
            "model_version": model_version,
            "type_source": type_source,
            "reason": reason,
            "message": message,
            "comparison_label": comparison_label,
            "comparison_scam_type": comparison_scam_type,
            "comparison_risk_score": comparison_risk_score,
            "comparison_model_confidence": comparison_model_confidence,
            "comparison_model_source": comparison_model_source,
            "comparison_model_version": comparison_model_version,
            "comparison_type_source": comparison_type_source,
            "comparison_tertiary_label": comparison_tertiary_label,
            "comparison_tertiary_scam_type": comparison_tertiary_scam_type,
            "comparison_tertiary_risk_score": comparison_tertiary_risk_score,
            "comparison_tertiary_model_confidence": comparison_tertiary_model_confidence,
            "comparison_tertiary_model_source": comparison_tertiary_model_source,
            "comparison_tertiary_model_version": comparison_tertiary_model_version,
            "comparison_tertiary_type_source": comparison_tertiary_type_source,
            "review_recommended": review_recommended,
            "review_reason": review_reason,
            "final_score_method": final_score_method,
            "model_outputs_json": model_outputs_json,
        }
    )
    rows = _request("POST", "scans", record)
    if isinstance(rows, list) and rows:
        return rows[0].get("id")
    return None


def read_scans(limit=5000):
    rows = _select("scans", SCAN_COLUMNS, limit=limit)
    return [_row_tuple(row, SCAN_COLUMNS) for row in rows]


def insert_feedback(
    language,
    message,
    pred_label,
    pred_scam_type,
    pred_risk_score,
    pred_confidence,
    model_source,
    model_version,
    feedback_type="wrong_prediction",
    expected_label=None,
    expected_scam_type=None,
    feedback_note=None,
):
    record = _clean_feedback_record(
        {
            "language": language,
            "message": message,
            "pred_label": pred_label,
            "pred_scam_type": pred_scam_type,
            "pred_risk_score": pred_risk_score,
            "pred_confidence": pred_confidence,
            "model_source": model_source,
            "model_version": model_version,
            "feedback_type": feedback_type,
            "expected_label": expected_label,
            "expected_scam_type": expected_scam_type,
            "feedback_note": feedback_note,
        }
    )
    rows = _request("POST", "feedback", record)
    if isinstance(rows, list) and rows:
        return rows[0].get("id")
    return None


def read_feedback(limit=2000):
    rows = _select("feedback", FEEDBACK_COLUMNS, limit=limit)
    return [_row_tuple(row, FEEDBACK_COLUMNS) for row in rows]


def read_feedback_summary():
    rows = _select("feedback", ["id", "feedback_type", "model_version"], limit=10000)
    wrong = sum(1 for row in rows if row.get("feedback_type") == "wrong_prediction")
    by_model_version: dict[str, int] = {}
    for row in rows:
        version = str(row.get("model_version") or "unknown")
        by_model_version[version] = by_model_version.get(version, 0) + 1
    return {
        "total_feedback": len(rows),
        "wrong_prediction": wrong,
        "by_model_version": by_model_version,
    }


def upsert_scan_records(records: list[dict[str, Any]], batch_size: int = 200) -> int:
    total = 0
    for start in range(0, len(records), batch_size):
        batch = [_clean_scan_record(record) for record in records[start : start + batch_size]]
        _request(
            "POST",
            "scans?on_conflict=id",
            batch,
            prefer="resolution=merge-duplicates,return=minimal",
        )
        total += len(batch)
    return total


def upsert_feedback_records(records: list[dict[str, Any]], batch_size: int = 200) -> int:
    total = 0
    for start in range(0, len(records), batch_size):
        batch = [_clean_feedback_record(record) for record in records[start : start + batch_size]]
        _request(
            "POST",
            "feedback?on_conflict=id",
            batch,
            prefer="resolution=merge-duplicates,return=minimal",
        )
        total += len(batch)
    return total


def reset_identity_sequences() -> None:
    """Move Supabase identity counters past migrated SQLite IDs."""
    _request("POST", "rpc/reset_safesandesh_identity_sequences", {}, prefer="")
