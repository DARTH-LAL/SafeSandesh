from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.auth import require_technical_password

os.environ["SAFESANDESH_APP_SHELL"] = "technical"


require_technical_password()

pages = [
    st.Page(
        ROOT / "pages" / "2_🧪_Technical_Lab.py",
        title="Dashboard",
        icon=":material/analytics:",
        url_path="",
        default=True,
    ),
    st.Page(
        ROOT / "pages" / "3_🧠_AI_Studio.py",
        title="AI Studio",
        icon=":material/neurology:",
        url_path="AI_Studio",
    ),
]

st.navigation(pages, position="hidden").run()
