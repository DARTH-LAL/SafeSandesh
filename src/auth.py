from __future__ import annotations

import hashlib
import hmac
import html
import os

import streamlit as st

from src.ui_theme import apply_theme, top_menu


def _secret_value(name: str) -> str:
    value = os.environ.get(name, "")
    if value:
        return value.strip()
    try:
        return str(st.secrets.get(name, "")).strip()
    except Exception:
        return ""


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _configured_password() -> tuple[str, str]:
    password_hash = _secret_value("TECHNICAL_APP_PASSWORD_HASH")
    if password_hash:
        return "hash", password_hash

    password = _secret_value("TECHNICAL_APP_PASSWORD")
    if password:
        return "plain", password

    return "", ""


def _password_matches(candidate: str, mode: str, expected: str) -> bool:
    if not candidate or not expected:
        return False
    if mode == "hash":
        return hmac.compare_digest(_hash_password(candidate), expected)
    return hmac.compare_digest(candidate, expected)


def _admin_menu_key() -> str:
    shell = os.environ.get("SAFESANDESH_APP_SHELL", "combined").strip().lower()
    return "admin" if shell == "consumer" else "dashboard"


def _apply_password_gate_theme() -> None:
    apply_theme(home_particles=True)
    top_menu(_admin_menu_key())


def _render_password_gate(message: str) -> None:
    safe_message = html.escape(message)
    _apply_password_gate_theme()
    st.markdown(
        f"""
        <style>
        .safesandesh-password-shell {{
          max-width: 1180px;
          margin: 4.5rem auto 0;
          padding: 0 1rem;
        }}
        .safesandesh-password-gate {{
          max-width: 980px;
          margin: 0 auto 1.25rem;
          padding: clamp(1.6rem, 4vw, 2.5rem);
          border: 1px solid rgba(0, 212, 255, 0.68);
          background:
            linear-gradient(135deg, rgba(0, 18, 30, 0.92), rgba(0, 8, 14, 0.94)),
            radial-gradient(circle at 18% 15%, rgba(0, 255, 159, 0.10), transparent 26rem);
          box-shadow:
            0 0 34px rgba(0, 212, 255, 0.24),
            inset 0 0 48px rgba(0, 255, 159, 0.045);
          backdrop-filter: blur(8px);
        }}
        .safesandesh-password-kicker {{
          color: #00ff9f;
          letter-spacing: 0.28em;
          text-transform: uppercase;
          font-weight: 800;
          font-size: 0.78rem;
        }}
        .safesandesh-password-gate h1 {{
          color: #f3fffb;
          letter-spacing: 0.14em;
          text-transform: uppercase;
          margin: 0.7rem 0;
        }}
        .safesandesh-password-gate p {{
          color: rgba(217, 255, 244, 0.78);
          line-height: 1.6;
          max-width: 760px;
        }}
        div[data-testid="stForm"] {{
          max-width: 1180px;
          margin: 0 auto;
          border: 1px solid rgba(0, 212, 255, 0.30);
          background: rgba(0, 8, 13, 0.72);
          box-shadow: 0 0 18px rgba(0, 212, 255, 0.12);
        }}
        div[data-testid="stAlert"] {{
          max-width: 1180px;
          margin-left: auto;
          margin-right: auto;
        }}
        </style>
        <div class="safesandesh-password-shell">
          <section class="safesandesh-password-gate">
            <div class="safesandesh-password-kicker">SafeSandesh Admin</div>
            <h1>Password Protected</h1>
            <p>{safe_message}</p>
          </section>
        </div>
        """,
        unsafe_allow_html=True,
    )


def require_technical_password() -> None:
    if _secret_value("SAFESANDESH_AUTH_DISABLED") == "1":
        return

    mode, expected = _configured_password()
    if not expected:
        _render_password_gate(
            "Admin access is disabled because no password is configured. "
            "Add TECHNICAL_APP_PASSWORD or TECHNICAL_APP_PASSWORD_HASH in Streamlit secrets."
        )
        st.stop()

    if st.session_state.get("technical_app_authenticated") is True:
        return

    _render_password_gate(
        "Enter the admin password to open the private dashboard, AI Studio, model comparisons, and stored scan analytics."
    )
    with st.form("technical_password_form"):
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Open Admin Area")

    if submitted:
        if _password_matches(password, mode, expected):
            st.session_state["technical_app_authenticated"] = True
            st.rerun()
        st.error("Incorrect password.")

    st.stop()
