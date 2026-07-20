from __future__ import annotations

import html
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from textwrap import dedent

import pandas as pd
import streamlit as st

from src.db import init_db, read_scans
from src.ui_theme import apply_theme, top_menu


SCAN_ROW_FIELDS = [
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

VERDICT_COLORS = {
    "phishing": "#ff3860",
    "suspicious": "#ffdd57",
    "safe": "#00ff9f",
}

LANGUAGE_COLORS = {
    "English": "#00d4ff",
    "Hindi": "#ff8c42",
    "Punjabi": "#b57aff",
    "Urdu": "#00ff9f",
}

SCAM_COLORS = [
    "#ff3860",
    "#ff8c42",
    "#ffdd57",
    "#00d4ff",
    "#00ff9f",
    "#b57aff",
    "#8ce6ff",
]


def _utc_now_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _render_html_block(raw: str) -> str:
    cleaned = dedent(raw).strip("\n")
    return "\n".join(line.lstrip() for line in cleaned.splitlines()).strip()


def _technical_dashboard_css() -> str:
    """Reuse the exact dashboard stylesheet from the technical dashboard."""
    technical_page = Path(__file__).with_name("2_🧪_Technical_Lab.py")
    try:
        source = technical_page.read_text(encoding="utf-8")
    except OSError:
        return ""

    matches = re.findall(r"<style>\s*\.dash-term-wrap.*?</style>", source, flags=re.DOTALL)
    return matches[-1] if matches else ""


def _query_param_value(name: str, default: str) -> str:
    value = st.query_params.get(name, default)
    if isinstance(value, list):
        value = value[0] if value else default
    return str(value or default).strip()


def _sanitize_period(value: str) -> str:
    value = str(value or "").strip().lower()
    return value if value in {"24h", "7d", "30d", "all"} else "24h"


def _sanitize_language(value: str) -> str:
    value = str(value or "").strip().lower()
    return value if value in {"all", "english", "hindi", "punjabi", "urdu"} else "all"


def _normalize_label(value: str) -> str:
    label = str(value or "").strip().lower()
    if "phish" in label:
        return "phishing"
    if "susp" in label or "spam" in label:
        return "suspicious"
    if "safe" in label or "ham" in label or "legit" in label:
        return "safe"
    return "unknown"


def _language_name(value: str) -> str:
    lang = str(value or "").strip().lower()
    mapping = {
        "en": "English",
        "english": "English",
        "hi": "Hindi",
        "hindi": "Hindi",
        "pa": "Punjabi",
        "punjabi": "Punjabi",
        "ur": "Urdu",
        "urdu": "Urdu",
    }
    return mapping.get(lang, str(value or "Unknown").strip() or "Unknown")


def _parse_scan_timestamps(values: pd.Series) -> pd.Series:
    """Parse both legacy SQLite timestamps and Supabase UTC timestamps."""
    try:
        parsed = pd.to_datetime(values, errors="coerce", utc=True, format="mixed")
    except (TypeError, ValueError):
        parsed = pd.to_datetime(values, errors="coerce", utc=True)
    return parsed.dt.tz_convert(None)


def _as_df(rows: list[tuple]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=SCAN_ROW_FIELDS)

    normalized = []
    for row in rows:
        values = list(row or [])
        values.extend([None] * max(0, len(SCAN_ROW_FIELDS) - len(values)))
        normalized.append(values[: len(SCAN_ROW_FIELDS)])

    df = pd.DataFrame(normalized, columns=SCAN_ROW_FIELDS)
    df["risk_score"] = pd.to_numeric(df["risk_score"], errors="coerce").fillna(0).astype(int)
    df["ts"] = _parse_scan_timestamps(df["ts"])
    df["label_norm"] = df["label"].map(_normalize_label)
    df["language_name"] = df["language"].map(_language_name)
    df["scam_type_clean"] = df["scam_type"].fillna("Other").astype(str).replace({"": "Other"})
    return df


def _apply_filters(df: pd.DataFrame, period_key: str, lang_key: str) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    if period_key in {"24h", "7d", "30d"}:
        now = pd.Timestamp(_utc_now_naive())
        if period_key == "24h":
            cutoff = now - pd.Timedelta(hours=24)
        elif period_key == "7d":
            cutoff = now - pd.Timedelta(days=7)
        else:
            cutoff = now - pd.Timedelta(days=30)
        out = out[out["ts"].notna() & (out["ts"] >= cutoff)]

    if lang_key != "all":
        out = out[out["language_name"].str.lower() == lang_key]

    return out


def _period_window(period_key: str) -> tuple[pd.Timedelta, str]:
    if period_key == "24h":
        return pd.Timedelta(hours=24), "vs prev 24h"
    if period_key == "7d":
        return pd.Timedelta(days=7), "vs prev 7d"
    if period_key == "30d":
        return pd.Timedelta(days=30), "vs prev 30d"
    return pd.Timedelta(days=7), "vs last week"


def _snapshot(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {"total": 0, "phishing": 0, "suspicious": 0, "safe": 0}
    counts = df["label_norm"].value_counts()
    return {
        "total": float(len(df)),
        "phishing": float(counts.get("phishing", 0)),
        "suspicious": float(counts.get("suspicious", 0)),
        "safe": float(counts.get("safe", 0)),
    }


def _trend(curr: float, prev: float, context: str) -> dict[str, str]:
    if prev <= 0:
        pct = 100.0 if curr > 0 else 0.0
    else:
        pct = ((curr - prev) / prev) * 100.0

    if pct > 0.05:
        return {"cls": "up", "arrow": "↑", "value": f"{pct:+.1f}%", "context": context}
    if pct < -0.05:
        return {"cls": "down", "arrow": "↓", "value": f"{pct:+.1f}%", "context": context}
    return {"cls": "flat", "arrow": "→", "value": "+0.0%", "context": context}


def _build_stat_trends(df: pd.DataFrame, period_key: str, lang_key: str) -> dict[str, dict[str, str]]:
    delta, context = _period_window(period_key)
    empty = {"cls": "flat", "arrow": "→", "value": "+0.0%", "context": context}
    if df.empty:
        return {key: dict(empty) for key in ["total", "phishing", "suspicious", "safe"]}

    base = df.copy()
    if lang_key != "all":
        base = base[base["language_name"].str.lower() == lang_key]

    now = pd.Timestamp(_utc_now_naive())
    current = base[(base["ts"] >= now - delta) & (base["ts"] <= now)]
    previous = base[(base["ts"] >= now - (delta * 2)) & (base["ts"] < now - delta)]
    curr = _snapshot(current)
    prev = _snapshot(previous)

    return {key: _trend(curr[key], prev[key], context) for key in ["total", "phishing", "suspicious", "safe"]}


def _timeline_from_df(df: pd.DataFrame) -> dict[str, list]:
    end = _utc_now_naive().date()
    days = [end - timedelta(days=i) for i in range(6, -1, -1)]
    labels = [day.strftime("%a") for day in days]

    if df.empty:
        return {"labels": labels, "phishing": [0] * 7, "suspicious": [0] * 7, "safe": [0] * 7}

    dfx = df[df["ts"].notna()].copy()
    dfx["day"] = dfx["ts"].dt.date

    def counts_for(label: str) -> list[int]:
        return [int(((dfx["day"] == day) & (dfx["label_norm"] == label)).sum()) for day in days]

    return {
        "labels": labels,
        "phishing": counts_for("phishing"),
        "suspicious": counts_for("suspicious"),
        "safe": counts_for("safe"),
    }


def _build_timeline_svg(labels: list[str], phishing: list[int], suspicious: list[int], safe: list[int]) -> str:
    width, height = 760, 210
    pad_l, pad_r, pad_t, pad_b = 44, 18, 18, 34
    chart_w, chart_h = width - pad_l - pad_r, height - pad_t - pad_b
    max_val = max(phishing + suspicious + safe + [1])

    def points(values: list[int]) -> list[tuple[float, float]]:
        out = []
        denominator = max(len(values) - 1, 1)
        for idx, value in enumerate(values):
            x = pad_l + (idx / denominator) * chart_w
            y = pad_t + chart_h - (float(value) / float(max_val)) * chart_h
            out.append((x, y))
        return out

    def polyline(values: list[int]) -> str:
        return " ".join(f"{x:.2f},{y:.2f}" for x, y in points(values))

    def area(values: list[int]) -> str:
        pts = points(values)
        if not pts:
            return ""
        start = f"M {pts[0][0]:.2f} {pts[0][1]:.2f}"
        middle = " ".join(f"L {x:.2f} {y:.2f}" for x, y in pts[1:])
        close = f"L {pts[-1][0]:.2f} {pad_t + chart_h:.2f} L {pts[0][0]:.2f} {pad_t + chart_h:.2f} Z"
        return f"{start} {middle} {close}"

    grid = []
    for idx in range(5):
        y = pad_t + chart_h - (idx / 4) * chart_h
        grid.append(f"<line x1='{pad_l}' y1='{y:.2f}' x2='{pad_l + chart_w}' y2='{y:.2f}' class='tl-grid' />")

    xlabels = []
    for idx, label in enumerate(labels):
        x = pad_l + (idx / max(len(labels) - 1, 1)) * chart_w
        xlabels.append(f"<text x='{x:.2f}' y='{height - 10}' text-anchor='middle' class='tl-label'>{html.escape(label)}</text>")

    def dots(values: list[int], class_name: str) -> str:
        return "".join(f"<circle cx='{x:.2f}' cy='{y:.2f}' r='2.9' class='{class_name}' />" for x, y in points(values))

    return (
        _render_html_block(
            f"""
            <svg class='dash-svg' viewBox='0 0 {width} {height}' preserveAspectRatio='none' role='img' aria-label='scans timeline'>
              <defs>
                <linearGradient id='consumer-fill-red' x1='0' x2='0' y1='0' y2='1'>
                  <stop offset='0%' stop-color='rgba(255,56,96,0.22)'/><stop offset='100%' stop-color='rgba(255,56,96,0)'/>
                </linearGradient>
                <linearGradient id='consumer-fill-yellow' x1='0' x2='0' y1='0' y2='1'>
                  <stop offset='0%' stop-color='rgba(255,221,87,0.18)'/><stop offset='100%' stop-color='rgba(255,221,87,0)'/>
                </linearGradient>
                <linearGradient id='consumer-fill-green' x1='0' x2='0' y1='0' y2='1'>
                  <stop offset='0%' stop-color='rgba(0,255,159,0.2)'/><stop offset='100%' stop-color='rgba(0,255,159,0)'/>
                </linearGradient>
              </defs>
              {''.join(grid)}
              <path d='{area(phishing)}' fill='url(#consumer-fill-red)'/>
              <path d='{area(suspicious)}' fill='url(#consumer-fill-yellow)'/>
              <path d='{area(safe)}' fill='url(#consumer-fill-green)'/>
              <polyline points='{polyline(phishing)}' class='tl-line-red' />
              <polyline points='{polyline(suspicious)}' class='tl-line-yellow' />
              <polyline points='{polyline(safe)}' class='tl-line-green' />
              {dots(phishing, 'tl-dot-red')}
              {dots(suspicious, 'tl-dot-yellow')}
              {dots(safe, 'tl-dot-green')}
              {''.join(xlabels)}
            </svg>
            """
        )
        .replace("\n", "")
        .strip()
    )


def _count_items(df: pd.DataFrame, column: str, limit: int | None = None) -> list[tuple[str, int]]:
    if df.empty or column not in df:
        return []
    counts = df[column].fillna("Unknown").astype(str).replace({"": "Unknown"}).value_counts()
    if limit:
        counts = counts.head(limit)
    return [(str(name), int(count)) for name, count in counts.items()]


def _donut_conic(items: list[tuple[str, int]], color_lookup: dict[str, str] | None = None, palette: list[str] | None = None) -> str:
    total = sum(count for _, count in items)
    if total <= 0:
        return "rgba(200,240,224,0.10) 0deg 360deg"

    segments = []
    start = 0.0
    for idx, (name, count) in enumerate(items):
        if count <= 0:
            continue
        color = (color_lookup or {}).get(name) or (color_lookup or {}).get(name.lower()) or (palette or SCAM_COLORS)[idx % len(palette or SCAM_COLORS)]
        end = start + (count / total) * 360.0
        segments.append(f"{color} {start:.2f}deg {end:.2f}deg")
        start = end
    return ", ".join(segments) or "rgba(200,240,224,0.10) 0deg 360deg"


def _legend_html(items: list[tuple[str, int]], color_lookup: dict[str, str] | None = None, palette: list[str] | None = None) -> str:
    total = sum(count for _, count in items)
    if total <= 0:
        return "<div class='log-empty'>No scan data yet</div>"

    rows = []
    for idx, (name, count) in enumerate(items):
        color = (color_lookup or {}).get(name) or (color_lookup or {}).get(name.lower()) or (palette or SCAM_COLORS)[idx % len(palette or SCAM_COLORS)]
        pct = (count / total) * 100.0
        rows.append(
            f"""
            <div class='legend-item'>
              <span class='legend-dot' style='background:{color};box-shadow:0 0 8px {color};'></span>
              <span class='legend-name'>{html.escape(name.title() if name.lower() in VERDICT_COLORS else name)}</span>
              <span class='legend-val' style='color:{color};'>{count:,}</span>
              <span class='legend-pct'>({pct:.0f}%)</span>
            </div>
            """
        )
    return _render_html_block("".join(rows))


def _hbar_rows(items: list[tuple[str, int]], color_lookup: dict[str, str] | None = None, palette: list[str] | None = None) -> str:
    if not items:
        return "<div class='log-empty'>No scan data yet</div>"
    max_count = max(count for _, count in items) or 1
    rows = []
    for idx, (name, count) in enumerate(items):
        color = (color_lookup or {}).get(name) or (color_lookup or {}).get(name.lower()) or (palette or SCAM_COLORS)[idx % len(palette or SCAM_COLORS)]
        width = max(4.0, (count / max_count) * 100.0)
        rows.append(
            f"""
            <div class='hbar-row'>
              <div class='hbar-top'><span class='hbar-name'>{html.escape(name)}</span><span class='hbar-val'>{count:,}</span></div>
              <div class='hbar-track'><div class='hbar-fill' style='width:{width:.1f}%;background:{color};box-shadow:0 0 12px {color};'></div></div>
            </div>
            """
        )
    return _render_html_block("".join(rows))


def _download_frame(
    *,
    period: str,
    language: str,
    total_scans: int,
    phishing_count: int,
    suspicious_count: int,
    safe_count: int,
    verdict_items: list[tuple[str, int]],
    language_items: list[tuple[str, int]],
    scam_items: list[tuple[str, int]],
) -> pd.DataFrame:
    """Public export: aggregate dashboard data only, never raw messages."""
    rows = [
        {"section": "filters", "metric": "period", "value": period},
        {"section": "filters", "metric": "language", "value": language},
        {"section": "summary", "metric": "total_scans", "value": total_scans},
        {"section": "summary", "metric": "phishing_detected", "value": phishing_count},
        {"section": "summary", "metric": "suspicious", "value": suspicious_count},
        {"section": "summary", "metric": "safe_messages", "value": safe_count},
    ]
    rows.extend({"section": "verdict_breakdown", "metric": name, "value": count} for name, count in verdict_items)
    rows.extend({"section": "language_distribution", "metric": name, "value": count} for name, count in language_items)
    rows.extend({"section": "scam_type_distribution", "metric": name, "value": count} for name, count in scam_items)
    return pd.DataFrame(rows, columns=["section", "metric", "value"])


st.set_page_config(page_title="Consumer Dashboard", layout="wide", initial_sidebar_state="collapsed")
init_db()
apply_theme(home_particles=True)
top_menu("dashboard")

terminal_time = datetime.now().strftime("%I:%M:%S %p").lower()
st.markdown(
    _render_html_block(
        f"""
        <div class='dash-term-wrap'>
          <div class='dash-term-line'>
            <span class='dash-term-prompt'>root@safesandesh:~$</span>
            <span class='dash-term-cmd'>consumer-dashboard --aggregate --privacy-safe --live</span>
            <span class='dash-term-sep'>|</span>
            <span class='dash-term-live'><span class='dash-live-dot'></span>LIVE DATA</span>
            <span class='dash-term-sep'>|</span>
            <span class='dash-term-label'>last_updated:</span>
            <span class='dash-term-time'>{html.escape(terminal_time)}</span>
            <span class='dash-term-block-inline'></span>
          </div>
        </div>
        """
    ),
    unsafe_allow_html=True,
)

period_selected = _sanitize_period(_query_param_value("period", "7d"))
lang_selected = _sanitize_language(_query_param_value("lang", "all"))

rows = read_scans(limit=6000)
df_all = _as_df(rows)
df = _apply_filters(df_all, period_selected, lang_selected)

counts = df["label_norm"].value_counts() if not df.empty else pd.Series(dtype=int)
total_scans = int(len(df))
phishing_count = int(counts.get("phishing", 0))
suspicious_count = int(counts.get("suspicious", 0))
safe_count = int(counts.get("safe", 0))
language_count = int(df["language_name"].nunique()) if not df.empty else 0

stat_trends = _build_stat_trends(df_all, period_selected, lang_selected)
timeline = _timeline_from_df(df)
verdict_items = [("phishing", phishing_count), ("suspicious", suspicious_count), ("safe", safe_count)]
language_items = _count_items(df, "language_name")
scam_items = _count_items(df, "scam_type_clean", limit=7)

period_tag_map = {
    "24h": "24-hour window",
    "7d": "7-day window",
    "30d": "30-day window",
    "all": "all time",
}
language_tag_map = {
    "all": "all languages",
    "english": "English",
    "hindi": "Hindi",
    "punjabi": "Punjabi",
    "urdu": "Urdu",
}
selected_period_tag = period_tag_map.get(period_selected, "24-hour window")
selected_language_tag = language_tag_map.get(lang_selected, "all languages")
selected_context_tag = f"{selected_period_tag} · {selected_language_tag}"

csv_bytes = b""
export_df = _download_frame(
    period=selected_period_tag,
    language=selected_language_tag,
    total_scans=total_scans,
    phishing_count=phishing_count,
    suspicious_count=suspicious_count,
    safe_count=safe_count,
    verdict_items=verdict_items,
    language_items=language_items,
    scam_items=scam_items,
)
csv_bytes = export_df.to_csv(index=False).encode("utf-8")

st.markdown(
    _render_html_block(
        """
        <style>
        :root {
          --consumer-cyan:#00d4ff;
          --consumer-green:#00ff9f;
          --consumer-red:#ff3860;
          --consumer-yellow:#ffdd57;
          --consumer-purple:#b57aff;
          --consumer-orange:#ff8c42;
          --consumer-panel:rgba(3,15,26,0.92);
          --consumer-border:rgba(0,255,159,0.20);
          --consumer-text:#e9f7fb;
          --consumer-muted:rgba(200,240,224,0.72);
        }
        .dash-wrap {
          border:1px solid rgba(0,212,255,0.38);
          background:
            linear-gradient(135deg, rgba(0,212,255,0.08), rgba(0,255,159,0.045)),
            rgba(2,12,22,0.72);
          box-shadow:0 0 20px rgba(0,212,255,0.16), inset 0 0 0 1px rgba(0,212,255,0.06);
          margin:1.15rem 0;
          padding:1.2rem;
        }
        .head-only { padding:1.55rem 1.75rem; }
        .main-only { border-color:rgba(0,255,159,0.25); box-shadow:0 0 18px rgba(0,255,159,0.08); }
        .dash-system-row { display:flex; justify-content:flex-end; margin-bottom:1.05rem; }
        .status-bar {
          display:inline-flex;
          align-items:center;
          gap:0.55rem;
          border:1px solid rgba(0,255,159,0.35);
          background:rgba(0,0,0,0.34);
          padding:0.36rem 0.7rem;
          font-family:'Share Tech Mono', monospace;
          color:var(--consumer-green);
          text-transform:uppercase;
          letter-spacing:0.12em;
          font-size:0.72rem;
        }
        .status-dot { width:7px; height:7px; border-radius:50%; background:var(--consumer-green); box-shadow:0 0 10px var(--consumer-green); }
        .version-tag { color:rgba(200,240,224,0.58); }
        .page-title-row { display:flex; align-items:baseline; gap:0.85rem; flex-wrap:wrap; }
        .section-code {
          color:var(--consumer-green);
          font-family:'Share Tech Mono', monospace;
          font-weight:900;
          letter-spacing:0.16em;
        }
        .page-h1 {
          color:#ffffff !important;
          -webkit-text-fill-color:#ffffff !important;
          font-size:clamp(2.15rem, 5.5vw, 5.1rem);
          line-height:0.9;
          text-transform:uppercase;
          letter-spacing:0.18em;
          margin:0.15rem 0 0.72rem;
          text-shadow:0 0 18px rgba(255,255,255,0.34), 0 0 34px rgba(0,212,255,0.18);
        }
        .page-sub {
          max-width:980px;
          color:rgba(221,246,239,0.84);
          font-size:1rem;
          line-height:1.72;
          margin:0;
        }
        .dash-filter-label {
          font-family:'Share Tech Mono', monospace;
          color:var(--consumer-cyan);
          text-transform:uppercase;
          letter-spacing:0.14em;
          font-size:0.62rem;
          margin-bottom:0.38rem;
          font-weight:900;
        }
        .stat-grid { display:grid; grid-template-columns:repeat(4, minmax(0,1fr)); gap:1.18rem; margin-bottom:1.95rem; }
        .stat-card {
          background:rgba(2,12,22,0.96);
          border:1px solid var(--consumer-border);
          border-radius:12px;
          padding:0.9rem 1rem;
          min-height:128px;
          position:relative;
          overflow:hidden;
          display:flex;
          flex-direction:column;
          justify-content:center;
          gap:0.38rem;
          transition:box-shadow .2s ease, border-color .2s ease, transform .2s ease;
        }
        .stat-card::after {
          content:'';
          position:absolute;
          left:0;
          right:0;
          bottom:0;
          height:1px;
          background:linear-gradient(90deg, transparent, var(--stat-glow, var(--consumer-green)), transparent);
          opacity:0.62;
        }
        .stat-card:hover { transform:translateY(-1px); }
        .stat-card.cyan { --stat-glow:var(--consumer-cyan); border-color:rgba(0,212,255,0.72); box-shadow:0 0 0 1px rgba(0,212,255,0.24),0 0 14px rgba(0,212,255,0.48),0 0 30px rgba(0,212,255,0.18),inset 0 0 0 1px rgba(0,212,255,0.10); }
        .stat-card.red { --stat-glow:var(--consumer-red); border-color:rgba(255,56,96,0.72); box-shadow:0 0 0 1px rgba(255,56,96,0.24),0 0 14px rgba(255,56,96,0.48),0 0 30px rgba(255,56,96,0.18),inset 0 0 0 1px rgba(255,56,96,0.10); }
        .stat-card.yellow { --stat-glow:var(--consumer-yellow); border-color:rgba(255,221,87,0.72); box-shadow:0 0 0 1px rgba(255,221,87,0.24),0 0 14px rgba(255,221,87,0.48),0 0 30px rgba(255,221,87,0.18),inset 0 0 0 1px rgba(255,221,87,0.10); }
        .stat-card.green { --stat-glow:var(--consumer-green); border-color:rgba(0,255,159,0.72); box-shadow:0 0 0 1px rgba(0,255,159,0.24),0 0 14px rgba(0,255,159,0.48),0 0 30px rgba(0,255,159,0.18),inset 0 0 0 1px rgba(0,255,159,0.10); }
        .stat-label {
          font-family:'Share Tech Mono', monospace;
          font-size:0.68rem;
          color:rgba(221,246,239,0.75);
          text-transform:uppercase;
          letter-spacing:0.18em;
          font-weight:900;
        }
        .stat-val {
          font-family:'Share Tech Mono', monospace;
          font-size:2.2rem;
          line-height:1;
          font-weight:1000;
        }
        .stat-val.cyan { color:var(--consumer-cyan); text-shadow:0 0 12px rgba(0,212,255,0.42); }
        .stat-val.red { color:var(--consumer-red); text-shadow:0 0 12px rgba(255,56,96,0.42); }
        .stat-val.yellow { color:var(--consumer-yellow); text-shadow:0 0 12px rgba(255,221,87,0.36); }
        .stat-val.green { color:var(--consumer-green); text-shadow:0 0 12px rgba(0,255,159,0.38); }
        .stat-trend {
          display:flex;
          align-items:center;
          gap:0.25rem;
          flex-wrap:wrap;
          font-family:'Share Tech Mono', monospace;
          font-size:0.68rem;
          font-weight:800;
          letter-spacing:0.04em;
        }
        .stat-trend .trend-value { font-weight:900; }
        .stat-trend .trend-context { opacity:0.92; font-weight:700; }
        .stat-trend.up { color:#11efad; text-shadow:0 0 8px rgba(17,239,173,0.28); }
        .stat-trend.down { color:#ff5d8a; text-shadow:0 0 8px rgba(255,93,138,0.28); }
        .stat-trend.flat { color:#b9d2db; text-shadow:0 0 8px rgba(185,210,219,0.20); }
        .grid-2-1 { display:grid; grid-template-columns:2fr 1fr; gap:1.4rem; margin-bottom:1.6rem; }
        .grid-2 { display:grid; grid-template-columns:1fr 1fr; gap:1.4rem; margin-bottom:1.6rem; }
        .panel {
          background:var(--consumer-panel);
          border:1px solid var(--consumer-border);
          border-radius:12px;
          position:relative;
          overflow:hidden;
          transition:box-shadow .2s ease, border-color .2s ease, transform .2s ease;
        }
        .panel:hover { transform:translateY(-1px); }
        .accent-cyan { border-color:rgba(0,212,255,0.72); box-shadow:0 0 0 1px rgba(0,212,255,0.24),0 0 14px rgba(0,212,255,0.48),0 0 30px rgba(0,212,255,0.18),inset 0 0 0 1px rgba(0,212,255,0.10); }
        .accent-red { border-color:rgba(255,56,96,0.72); box-shadow:0 0 0 1px rgba(255,56,96,0.24),0 0 14px rgba(255,56,96,0.48),0 0 30px rgba(255,56,96,0.18),inset 0 0 0 1px rgba(255,56,96,0.10); }
        .accent-yellow { border-color:rgba(255,221,87,0.72); box-shadow:0 0 0 1px rgba(255,221,87,0.24),0 0 14px rgba(255,221,87,0.48),0 0 30px rgba(255,221,87,0.18),inset 0 0 0 1px rgba(255,221,87,0.10); }
        .accent-purple { border-color:rgba(181,122,255,0.74); box-shadow:0 0 0 1px rgba(181,122,255,0.24),0 0 14px rgba(181,122,255,0.48),0 0 30px rgba(181,122,255,0.18),inset 0 0 0 1px rgba(181,122,255,0.10); }
        .accent-green { border-color:rgba(0,255,159,0.72); box-shadow:0 0 0 1px rgba(0,255,159,0.24),0 0 14px rgba(0,255,159,0.48),0 0 30px rgba(0,255,159,0.18),inset 0 0 0 1px rgba(0,255,159,0.10); }
        .panel-head {
          display:flex;
          align-items:center;
          justify-content:space-between;
          gap:0.7rem;
          padding:0.82rem 1.08rem;
          border-bottom:1px solid var(--consumer-border);
          background:rgba(0,0,0,.24);
        }
        .panel-title {
          font-family:'Share Tech Mono', monospace;
          font-size:0.9rem;
          font-weight:900;
          color:var(--consumer-cyan);
          letter-spacing:0.12em;
          text-transform:uppercase;
          display:flex;
          align-items:center;
          gap:8px;
          text-shadow:0 0 8px rgba(0,212,255,0.24);
        }
        .panel-title::before { content:'//'; color:rgba(0,212,255,0.35); }
        .panel-tag {
          font-family:'Share Tech Mono', monospace;
          font-size:0.62rem;
          color:#d7eaf0;
          border:1px solid rgba(0,255,159,0.28);
          background:rgba(0,0,0,0.26);
          padding:0.18rem 0.5rem;
          text-transform:uppercase;
          letter-spacing:0.08em;
        }
        .panel-body { padding:1.12rem 1.08rem; }
        .tl-legend {
          display:flex;
          align-items:center;
          gap:1.15rem;
          padding:0.1rem 0.15rem 0.46rem;
          margin-bottom:0.48rem;
          border-bottom:1px solid rgba(0,255,159,0.18);
        }
        .tl-legend-item {
          display:inline-flex;
          align-items:center;
          gap:0.48rem;
          font-family:'Share Tech Mono', monospace;
          font-size:0.82rem;
          color:#d6e7ee;
        }
        .tl-legend-line { width:20px; height:3px; border-radius:99px; display:inline-block; }
        .tl-legend-line.red { background:var(--consumer-red); box-shadow:0 0 8px rgba(255,56,96,0.58); }
        .tl-legend-line.yellow { background:var(--consumer-yellow); box-shadow:0 0 8px rgba(255,221,87,0.58); }
        .tl-legend-line.green { background:var(--consumer-green); box-shadow:0 0 8px rgba(0,255,159,0.58); }
        .dash-svg { width:100%; height:210px; display:block; overflow:visible; }
        .tl-grid { stroke:rgba(0,255,159,0.09); stroke-width:1; }
        .tl-label { fill:rgba(200,240,224,0.48); font-family:'Share Tech Mono', monospace; font-size:11px; }
        .tl-line-red,.tl-line-yellow,.tl-line-green { fill:none; stroke-width:2.6; stroke-linejoin:round; stroke-linecap:round; }
        .tl-line-red { stroke:var(--consumer-red); filter:drop-shadow(0 0 5px rgba(255,56,96,0.50)); }
        .tl-line-yellow { stroke:var(--consumer-yellow); filter:drop-shadow(0 0 5px rgba(255,221,87,0.50)); }
        .tl-line-green { stroke:var(--consumer-green); filter:drop-shadow(0 0 5px rgba(0,255,159,0.50)); }
        .tl-dot-red { fill:var(--consumer-red); }
        .tl-dot-yellow { fill:var(--consumer-yellow); }
        .tl-dot-green { fill:var(--consumer-green); }
        .donut-wrap { display:flex; align-items:center; justify-content:center; gap:1.2rem; min-height:205px; }
        .donut { position:relative; width:160px; height:160px; border-radius:50%; box-shadow:0 0 20px rgba(0,212,255,0.08); }
        .donut::after { content:''; position:absolute; inset:24px; border-radius:50%; background:rgba(3,15,26,0.98); border:1px solid rgba(0,255,159,0.14); z-index:2; }
        .donut-center {
          position:absolute;
          inset:0;
          display:flex;
          align-items:center;
          justify-content:center;
          z-index:3;
          font-family:'Share Tech Mono', monospace;
          font-weight:900;
          color:#d8fff0;
          font-size:1.1rem;
        }
        .legend { display:flex; flex-direction:column; gap:0.48rem; min-width:210px; }
        .legend-item { display:flex; align-items:center; gap:0.45rem; }
        .legend-dot { width:8px; height:8px; border-radius:2px; }
        .legend-name {
          font-family:'Share Tech Mono', monospace;
          font-size:0.78rem;
          font-weight:700;
          color:#e9f7fb;
          flex:1;
        }
        .legend-val { font-family:'Share Tech Mono', monospace; font-size:0.84rem; font-weight:900; }
        .legend-pct { font-family:'Share Tech Mono', monospace; font-size:0.68rem; color:#c6dbe2; font-weight:700; }
        .hbar-list { display:flex; flex-direction:column; gap:0.75rem; }
        .hbar-top { display:flex; align-items:center; justify-content:space-between; gap:0.6rem; margin-bottom:0.24rem; }
        .hbar-name,.hbar-val { font-family:'Share Tech Mono', monospace; font-size:0.74rem; }
        .hbar-name { color:var(--consumer-text); }
        .hbar-val { color:var(--consumer-muted); }
        .hbar-track { height:7px; background:rgba(255,255,255,0.06); }
        .hbar-fill { height:100%; transform-origin:left center; }
        .log-empty { font-family:'Share Tech Mono', monospace; font-size:0.72rem; color:var(--consumer-muted); text-align:center; padding:1.2rem; letter-spacing:0.08em; }
        @media (max-width: 1200px) {
          .stat-grid { grid-template-columns: repeat(3, minmax(0,1fr)); }
          .grid-2-1, .grid-2 { grid-template-columns: 1fr; }
        }
        @media (max-width: 760px) {
          .stat-grid { grid-template-columns: repeat(2, minmax(0,1fr)); }
          .page-h1 { font-size: 1.65rem; }
          .donut-wrap { flex-direction:column; }
        }
        </style>
        """
    ),
    unsafe_allow_html=True,
)

st.markdown(_render_html_block(_technical_dashboard_css()), unsafe_allow_html=True)

st.markdown(
    _render_html_block(
        f"""
        <style>
        .dash-wrap {{
          border:0 !important;
          background:transparent !important;
          box-shadow:none !important;
        }}
        .dash-system-row .status-bar {{
          border:0 !important;
          background:transparent !important;
          padding:0 !important;
          box-shadow:none !important;
          color:inherit !important;
        }}
        .dash-system-row .status-dot {{
          width:9px !important;
          height:9px !important;
        }}
        .st-key-dash_period_{period_selected} .stButton > button,
        .st-key-dash_lang_{lang_selected} .stButton > button {{
          color:#00f1a8 !important;
          -webkit-text-fill-color:#00f1a8 !important;
          border-color:rgba(0,255,159,0.82) !important;
          background:rgba(0,0,0,0.90) !important;
          box-shadow:0 0 11px rgba(0,255,159,0.16), inset 0 0 0 1px rgba(0,255,159,0.12) !important;
        }}
        </style>
        """
    ),
    unsafe_allow_html=True,
)

st.markdown(
    _render_html_block(
        """
        <div class='dash-wrap head-only'>
          <div class='dash-system-row'>
            <div class='status-bar'>
              <div class='status-dot'></div>
              <span class='status-text'>System Online</span>
              <span class='version-tag'>consumer view</span>
            </div>
          </div>
          <div class='page-title-row'>
            <span class='section-code'>// 02</span>
            <h1 class='page-h1'>Consumer Dashboard</h1>
          </div>
          <p class='page-sub'>Privacy-safe user analytics for scan volume, language coverage, scam categories, verdict counts, and recent activity. Individual messages and model comparisons stay in the private analyst app.</p>
        </div>
        """
    ),
    unsafe_allow_html=True,
)

filters_left, filters_mid, filters_right = st.columns([4.5, 4.5, 2.6], gap="medium")
with filters_left:
    st.markdown("<div class='dash-filter-label'>Period:</div>", unsafe_allow_html=True)
    period_cols = st.columns(4, gap="small")
    for column, label, key_value in zip(period_cols, ["24H", "7D", "30D", "ALL TIME"], ["24h", "7d", "30d", "all"]):
        if column.button(label, key=f"dash_period_{key_value}", use_container_width=True):
            st.query_params["period"] = key_value
            st.query_params["lang"] = lang_selected
            st.rerun()

with filters_mid:
    st.markdown("<div class='dash-filter-label'>Language:</div>", unsafe_allow_html=True)
    lang_cols = st.columns(5, gap="small")
    for column, label, key_value in zip(
        lang_cols,
        ["All", "English", "Hindi", "Punjabi", "Urdu"],
        ["all", "english", "hindi", "punjabi", "urdu"],
    ):
        if column.button(label, key=f"dash_lang_{key_value}", use_container_width=True):
            st.query_params["period"] = period_selected
            st.query_params["lang"] = key_value
            st.rerun()

with filters_right:
    st.markdown("<div class='dash-filter-label'>Export:</div>", unsafe_allow_html=True)
    st.download_button(
        label="Download Summary CSV",
        data=csv_bytes,
        file_name="consumer_dashboard_summary.csv",
        mime="text/csv",
        key="dash_export_csv",
        disabled=not bool(csv_bytes),
        use_container_width=True,
    )

st.markdown(
    _render_html_block(
        f"""
        <div class='dash-wrap main-only'>
          <div class='stat-grid'>
            <div class='stat-card cyan'>
              <div class='stat-label'>Total Scans</div>
              <div class='stat-val cyan'>{total_scans:,}</div>
              <div class='stat-trend {stat_trends['total']['cls']}'><span class='trend-arrow'>{stat_trends['total']['arrow']}</span><span class='trend-value'>{stat_trends['total']['value']}</span><span class='trend-context'>{html.escape(stat_trends['total']['context'])}</span></div>
            </div>
            <div class='stat-card red'>
              <div class='stat-label'>Phishing Detected</div>
              <div class='stat-val red'>{phishing_count:,}</div>
              <div class='stat-trend {stat_trends['phishing']['cls']}'><span class='trend-arrow'>{stat_trends['phishing']['arrow']}</span><span class='trend-value'>{stat_trends['phishing']['value']}</span><span class='trend-context'>{html.escape(stat_trends['phishing']['context'])}</span></div>
            </div>
            <div class='stat-card yellow'>
              <div class='stat-label'>Suspicious</div>
              <div class='stat-val yellow'>{suspicious_count:,}</div>
              <div class='stat-trend {stat_trends['suspicious']['cls']}'><span class='trend-arrow'>{stat_trends['suspicious']['arrow']}</span><span class='trend-value'>{stat_trends['suspicious']['value']}</span><span class='trend-context'>{html.escape(stat_trends['suspicious']['context'])}</span></div>
            </div>
            <div class='stat-card green'>
              <div class='stat-label'>Safe Messages</div>
              <div class='stat-val green'>{safe_count:,}</div>
              <div class='stat-trend {stat_trends['safe']['cls']}'><span class='trend-arrow'>{stat_trends['safe']['arrow']}</span><span class='trend-value'>{stat_trends['safe']['value']}</span><span class='trend-context'>{html.escape(stat_trends['safe']['context'])}</span></div>
            </div>
          </div>

          <div class='grid-2-1'>
            <div class='panel accent-cyan'>
              <div class='panel-head'><div class='panel-title'>Scans Over Time</div><span class='panel-tag'>{html.escape(selected_period_tag)}</span></div>
              <div class='panel-body'>
                <div class='tl-legend'>
                  <span class='tl-legend-item'><span class='tl-legend-line red'></span>Phishing</span>
                  <span class='tl-legend-item'><span class='tl-legend-line yellow'></span>Suspicious</span>
                  <span class='tl-legend-item'><span class='tl-legend-line green'></span>Safe</span>
                </div>
                {_build_timeline_svg(timeline['labels'], timeline['phishing'], timeline['suspicious'], timeline['safe'])}
              </div>
            </div>

            <div class='panel accent-red'>
              <div class='panel-head'><div class='panel-title'>Verdict Breakdown</div><span class='panel-tag'>{html.escape(selected_language_tag)}</span></div>
              <div class='panel-body'>
                <div class='donut-wrap'>
                  <div class='donut' style='background:conic-gradient({_donut_conic(verdict_items, color_lookup=VERDICT_COLORS)});'>
                    <div class='donut-center'>{total_scans:,}</div>
                  </div>
                  <div class='legend'>{_legend_html(verdict_items, color_lookup=VERDICT_COLORS)}</div>
                </div>
              </div>
            </div>
          </div>

          <div class='grid-2'>
            <div class='panel accent-yellow'>
              <div class='panel-head'><div class='panel-title'>Scam Type Distribution</div><span class='panel-tag'>{html.escape(selected_context_tag)}</span></div>
              <div class='panel-body'><div class='hbar-list'>{_hbar_rows(scam_items, palette=SCAM_COLORS)}</div></div>
            </div>

            <div class='panel accent-purple'>
              <div class='panel-head'><div class='panel-title'>Language Distribution</div><span class='panel-tag'>{language_count:,} languages visible</span></div>
              <div class='panel-body'>
                <div class='donut-wrap'>
                  <div class='donut' style='background:conic-gradient({_donut_conic(language_items, color_lookup=LANGUAGE_COLORS)});'>
                    <div class='donut-center'>{language_count:,}</div>
                  </div>
                  <div class='legend'>{_legend_html(language_items, color_lookup=LANGUAGE_COLORS)}</div>
                </div>
              </div>
            </div>
          </div>

        </div>
        """
    ),
    unsafe_allow_html=True,
)
