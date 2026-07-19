from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["SAFESANDESH_APP_SHELL"] = "consumer"
os.environ["SAFESANDESH_DETECTOR_PAGE_PATH"] = str(ROOT / "pages" / "1_🛡️_Detector.py")

pages = [
    st.Page(ROOT / "app.py", title="Home", icon=":material/home:", url_path="", default=True),
    st.Page(
        ROOT / "pages" / "1_🛡️_Detector.py",
        title="Detector",
        icon=":material/shield:",
        url_path="Detector",
    ),
    st.Page(
        ROOT / "pages" / "2_📈_Consumer_Dashboard.py",
        title="Dashboard",
        icon=":material/monitoring:",
        url_path="Dashboard",
    ),
    st.Page(
        ROOT / "pages" / "2_🧪_Technical_Lab.py",
        title="Analyst Lab",
        icon=":material/science:",
        url_path="Analyst_Lab",
    ),
    st.Page(
        ROOT / "pages" / "3_🧠_AI_Studio.py",
        title="AI Studio",
        icon=":material/neurology:",
        url_path="AI_Studio",
    ),
]

st.navigation(pages, position="hidden").run()
