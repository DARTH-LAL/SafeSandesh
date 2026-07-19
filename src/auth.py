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


def _protected_menu_key() -> str:
    shell = os.environ.get("SAFESANDESH_APP_SHELL", "combined").strip().lower()
    return "analyst_lab" if shell == "consumer" else "dashboard"


def _apply_password_gate_theme() -> None:
    apply_theme(home_particles=True)
    top_menu(_protected_menu_key())


def _render_password_gate(message: str = "") -> None:
    detail_html = ""
    if message:
        safe_message = html.escape(message)
        detail_html = f"<p class='safesandesh-password-detail'>{safe_message}</p>"
    _apply_password_gate_theme()
    st.markdown(
        f"""
        <style>
        .safesandesh-password-shell {{
          max-width: 1180px;
          margin: 3.75rem auto 0;
          padding: 0 1rem;
        }}
        .safesandesh-password-gate {{
          max-width: 940px;
          margin: 0 auto 1.25rem;
          border: 1px solid rgba(0, 212, 255, 0.72);
          background:
            linear-gradient(135deg, rgba(0, 17, 29, 0.94), rgba(0, 7, 13, 0.96)),
            radial-gradient(circle at 18% 15%, rgba(0, 255, 159, 0.10), transparent 26rem);
          box-shadow:
            0 0 34px rgba(0, 212, 255, 0.24),
            inset 0 0 48px rgba(0, 255, 159, 0.045);
          backdrop-filter: blur(8px);
          overflow: hidden;
        }}
        .safesandesh-password-titlebar {{
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 1rem;
          padding: 0.85rem 1.1rem;
          border-bottom: 1px solid rgba(0, 255, 159, 0.18);
          background: rgba(0, 0, 0, 0.28);
        }}
        .safesandesh-password-lights {{
          display: flex;
          gap: 0.5rem;
        }}
        .safesandesh-password-lights span {{
          width: 0.72rem;
          height: 0.72rem;
          border-radius: 999px;
          display: inline-block;
        }}
        .safesandesh-password-lights span:nth-child(1) {{
          background: #ff3860;
          box-shadow: 0 0 10px rgba(255, 56, 96, 0.85);
        }}
        .safesandesh-password-lights span:nth-child(2) {{
          background: #ffdd57;
          box-shadow: 0 0 10px rgba(255, 221, 87, 0.75);
        }}
        .safesandesh-password-lights span:nth-child(3) {{
          background: #00ff9f;
          box-shadow: 0 0 10px rgba(0, 255, 159, 0.75);
        }}
        .safesandesh-password-file {{
          color: rgba(200, 240, 224, 0.62);
          font-family: 'Share Tech Mono', monospace;
          font-size: 0.78rem;
          letter-spacing: 0.18em;
          text-transform: uppercase;
        }}
        .safesandesh-password-body {{
          padding: clamp(1.55rem, 4vw, 2.45rem);
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
          font-size: clamp(2rem, 5vw, 3.5rem);
          text-shadow: 0 0 20px rgba(255, 255, 255, 0.22);
        }}
        .safesandesh-password-detail {{
          color: rgba(217, 255, 244, 0.78);
          line-height: 1.6;
          max-width: 760px;
          margin: 0.6rem 0 0;
        }}
        div[data-testid="stForm"] {{
          max-width: 940px;
          margin: 0 auto;
          border: 1px solid rgba(0, 255, 159, 0.42);
          background:
            linear-gradient(135deg, rgba(0, 17, 29, 0.88), rgba(0, 7, 13, 0.94));
          box-shadow:
            0 0 24px rgba(0, 255, 159, 0.14),
            inset 0 0 32px rgba(0, 212, 255, 0.04);
          backdrop-filter: blur(8px);
        }}
        div[data-testid="stForm"] label p {{
          color: #d9fff4 !important;
          font-family: 'Share Tech Mono', monospace;
          letter-spacing: 0.08em;
          text-transform: uppercase;
        }}
        div[data-testid="stForm"] input {{
          border: 1px solid rgba(255, 56, 96, 0.7) !important;
          background: rgba(0, 0, 0, 0.55) !important;
          color: #f3fffb !important;
          box-shadow: inset 0 0 18px rgba(255, 56, 96, 0.08);
        }}
        div[data-testid="stForm"] button {{
          border: 1px solid rgba(0, 212, 255, 0.58) !important;
          background: rgba(0, 18, 30, 0.88) !important;
          color: #d9fff4 !important;
          box-shadow: 0 0 18px rgba(0, 212, 255, 0.12);
        }}
        div[data-testid="stForm"] button:hover {{
          border-color: rgba(0, 255, 159, 0.88) !important;
          color: #00ff9f !important;
          box-shadow: 0 0 24px rgba(0, 255, 159, 0.20);
        }}
        div[data-testid="stAlert"] {{
          max-width: 940px;
          margin-left: auto;
          margin-right: auto;
        }}
        </style>
        <div class="safesandesh-password-shell">
          <section class="safesandesh-password-gate">
            <div class="safesandesh-password-titlebar">
              <div class="safesandesh-password-lights"><span></span><span></span><span></span></div>
              <div class="safesandesh-password-file">analyst_lab.py - locked</div>
            </div>
            <div class="safesandesh-password-body">
              <div class="safesandesh-password-kicker">SafeSandesh Analyst Lab</div>
              <h1>Password Protected</h1>
              {detail_html}
            </div>
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
            "Analyst Lab access is disabled because no password is configured. "
            "Add TECHNICAL_APP_PASSWORD or TECHNICAL_APP_PASSWORD_HASH in Streamlit secrets."
        )
        st.stop()

    if st.session_state.get("technical_app_authenticated") is True:
        return

    _render_password_gate()
    with st.form("technical_password_form"):
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Open Analyst Lab")

    if submitted:
        if _password_matches(password, mode, expected):
            st.session_state["technical_app_authenticated"] = True
            st.rerun()
        st.error("Incorrect password.")

    st.stop()
