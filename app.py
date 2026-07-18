import random
import os
from pathlib import Path

import streamlit as st

from src.db import init_db
from src.ui_theme import apply_theme, chips, terminal_header, top_menu


SAMPLE_SCAM_MESSAGE = (
    "URGENT: Your bank KYC is pending. Account will be blocked today. "
    "Verify now: https://tinyurl.com/kyc-check and share OTP to avoid suspension."
)

APP_SHELL = os.environ.get("SAFESANDESH_APP_SHELL", "combined").strip().lower()


def _detector_page_target() -> str | Path:
    configured = os.environ.get("SAFESANDESH_DETECTOR_PAGE_PATH")
    if APP_SHELL == "consumer" and configured:
        return Path(configured)
    return "pages/1_🛡️_Detector.py"

INDIA_OUTLINE_PATH = (
    "M 150.4 44.2 L 163.3 58.3 L 162.1 68.2 L 166.9 74.3 L 166.5 80.5 L 157.8 78.9 "
    "L 161.2 92.1 L 173.0 99.8 L 189.8 108.2 L 182.1 113.6 L 177.5 124.9 L 189.1 129.5 "
    "L 200.5 135.4 L 216.2 142.1 L 232.7 143.7 L 239.7 149.8 L 249.0 151.0 L 263.5 153.8 "
    "L 273.5 153.6 L 274.9 148.8 L 273.3 141.2 L 274.2 136.0 L 281.6 133.4 L 282.6 142.9 "
    "L 282.8 145.3 L 293.8 149.9 L 301.3 148.0 L 311.5 148.8 L 321.3 148.5 L 322.2 141.1 "
    "L 317.3 137.2 L 327.0 135.7 L 338.0 126.8 L 351.8 119.1 L 361.9 122.1 L 370.5 117.0 "
    "L 376.2 124.5 L 372.1 129.5 L 385.1 131.3 L 386.0 135.9 L 381.8 138.1 L 382.8 145.5 "
    "L 374.2 143.3 L 358.6 151.7 L 358.9 158.6 L 352.3 168.7 L 351.7 174.5 L 346.3 184.5 "
    "L 336.9 181.7 L 336.4 194.2 L 333.7 198.3 L 335.0 203.4 L 329.0 206.2 L 322.7 187.1 "
    "L 319.4 187.2 L 317.4 194.9 L 310.8 188.6 L 314.5 181.8 L 319.9 181.1 L 325.5 170.9 "
    "L 318.5 168.8 L 307.3 169.0 L 295.9 167.4 L 294.8 159.0 L 289.1 158.4 L 279.5 153.2 "
    "L 275.3 161.4 L 284.0 167.7 L 276.4 172.2 L 273.8 176.6 L 281.2 179.8 L 279.1 187.1 "
    "L 283.3 196.2 L 285.2 206.1 L 283.5 210.5 L 275.3 210.3 L 260.4 212.8 L 261.1 221.9 "
    "L 254.7 229.0 L 237.3 237.1 L 223.9 251.3 L 214.8 258.9 L 202.8 266.8 L 202.8 272.3 "
    "L 196.8 275.3 L 185.9 279.6 L 180.3 280.2 L 176.7 289.4 L 179.2 305.1 L 179.8 315.1 "
    "L 174.7 326.5 L 174.7 347.0 L 168.5 347.6 L 163.0 356.7 L 166.6 360.7 L 155.7 364.1 "
    "L 151.6 372.3 L 146.8 375.8 L 135.4 364.5 L 129.8 347.7 L 125.2 335.5 L 121.0 329.8 "
    "L 114.6 318.3 L 111.6 303.2 L 109.5 295.7 L 98.5 279.1 L 93.5 255.8 L 89.9 240.4 "
    "L 90.0 225.8 L 87.6 214.5 L 70.1 221.7 L 61.6 220.3 L 45.9 205.7 L 51.7 201.3 "
    "L 48.1 196.6 L 34.0 186.4 L 42.0 178.3 L 68.5 178.4 L 66.1 168.0 L 59.4 161.9 "
    "L 58.0 152.6 L 50.1 147.2 L 63.4 134.6 L 77.4 135.5 L 90.0 122.9 L 97.5 110.7 "
    "L 109.2 98.6 L 109.0 90.0 L 119.3 83.0 L 109.6 77.1 L 105.4 68.9 L 101.1 58.4 "
    "L 107.0 53.2 L 125.3 56.1 L 138.7 54.3 L 150.4 44.2 Z"
)

ALERT_RED_POINTS = [
    (142, 126),  # Delhi
    (90, 242),   # Mumbai
    (277, 200),  # Kolkata
    (158, 262),  # Hyderabad
    (108, 182),
    (246, 176),
    (186, 226),
    (224, 146),
]

ALERT_ORANGE_POINTS = [
    (188, 148),  # Lucknow
    (180, 314),  # Chennai
    (318, 157),  # Guwahati
    (152, 346),
    (118, 148),
    (302, 178),
    (204, 286),
    (264, 228),
]

DEFENSE_GREEN_POINTS = [
    (148, 316),  # Bengaluru
    (126, 148),  # Jaipur
    (216, 212),
    (138, 286),
    (170, 172),
    (236, 238),
    (292, 170),
    (154, 96),
]

DEFENSE_CYAN_POINTS = [
    (131, 352),  # Kochi
    (223, 251),
    (114, 205),
    (260, 193),
    (248, 278),
    (334, 164),
    (103, 262),
    (186, 332),
]

MAP_LINK_PATHS = [
    "M90 242 Q151 171 277 200",
    "M142 126 Q209 135 318 157",
    "M158 262 Q168 298 180 314",
    "M126 148 Q156 142 188 148",
    "M108 182 Q176 172 302 178",
    "M138 286 Q176 238 246 176",
    "M114 205 Q182 226 264 228",
    "M154 96 Q218 134 334 164",
]


def _map_link_bits() -> str:
    return "".join([f"<path id='map-route-{idx}' class='india-link' d='{d}' />" for idx, d in enumerate(MAP_LINK_PATHS)])


def _map_packet_bits() -> str:
    bits: list[str] = []
    for route_idx in range(len(MAP_LINK_PATHS)):
        packet_count = 2 if route_idx % 2 == 0 else 1
        for packet_idx in range(packet_count):
            packet_class = "map-packet map-packet-cyan" if (route_idx + packet_idx) % 2 else "map-packet map-packet-neon"
            radius = 1.15 + (0.2 * (packet_idx % 2))
            duration = 8.6 + ((route_idx * 1.15 + packet_idx * 0.65) % 4.2)
            delay = (route_idx * 0.82 + packet_idx * 1.37) % 6.4
            bits.append(
                (
                    f"<circle class='{packet_class}' r='{radius:.2f}'>"
                    f"<animateMotion dur='{duration:.2f}s' begin='-{delay:.2f}s' repeatCount='indefinite' rotate='auto'>"
                    f"<mpath href='#map-route-{route_idx}' /></animateMotion></circle>"
                )
            )
    return "".join(bits)


def _map_node_bits(points: list[tuple[int, int]], node_class: str, ring_class: str, node_base: float, ring_base: float) -> str:
    bits: list[str] = []
    for idx, (x, y) in enumerate(points):
        node_r = node_base + ((idx % 3) * 0.15)
        ring_r = ring_base + ((idx % 4) * 0.16)
        delay = 0.24 + (idx * 0.23)
        burst_key = f"{node_class}-{idx}"
        bits.append(
            (
                f"<circle class='{ring_class} map-burst-ring' data-burst-key='{burst_key}' "
                f"cx='{x}' cy='{y}' r='{ring_r:.2f}' style='animation-delay:-{delay:.2f}s;' />"
            )
        )
        bits.append(
            f"<circle class='{node_class} map-burst-node' data-burst-key='{burst_key}' cx='{x}' cy='{y}' r='{node_r:.2f}' />"
        )
    return "".join(bits)


def _map_particle_bits() -> str:
    rng = random.Random()
    anchors = (
        ALERT_RED_POINTS
        + ALERT_ORANGE_POINTS
        + DEFENSE_GREEN_POINTS
        + DEFENSE_CYAN_POINTS
        + [(240, 230), (203, 280), (170, 300), (98, 181), (249, 151), (320, 187), (174, 326), (126, 335)]
    )

    bits: list[str] = []
    for x, y in anchors:
        for _ in range(rng.randint(3, 5)):
            dx = rng.uniform(-8.0, 8.0)
            dy = rng.uniform(-8.0, 8.0)
            radius = rng.uniform(0.7, 1.8)
            duration = rng.uniform(2.0, 4.6)
            delay = rng.uniform(0.0, 3.2)
            is_cyan = rng.random() > 0.62
            hue_class = "map-bit-cyan" if is_cyan else "map-bit-neon"
            fill = "rgba(64,198,255,0.82)" if is_cyan else "rgba(0,255,200,0.82)"
            bits.append(
                (
                    f"<circle class='map-bit {hue_class}' cx='{x + dx:.1f}' cy='{y + dy:.1f}' r='{radius:.2f}' fill='{fill}' "
                    f"style='fill:{fill}; animation-duration:{duration:.2f}s; animation-delay:-{delay:.2f}s;' />"
                )
            )
    return "".join(bits)


st.set_page_config(page_title="SafeSandesh", layout="wide", initial_sidebar_state="collapsed")
init_db()
apply_theme(home_particles=True)
top_menu("home")
terminal_header("initializing fraud detection engine... india_db_v2.4 loaded")

left, right = st.columns([1.05, 1.15], gap="large")

with left:
    st.markdown(
        """
        <div class="status-bar">
            <div class="status-dot"></div>
            <span class="status-text">System Online</span>
            <span class="version-tag">v2.4.1 — IND</span>
        </div>
        <h1 class="cyber-title">
            <span class="line1">Indian Mobile</span>
            <span class="line2">SafeSandesh</span>
            <span class="line3">Protection.Active</span>
        </h1>
        <p class="hero-sub">
            Preview the experience, then jump directly to scanning on Detector.
            AI-powered detection for OTP, KYC and UPI fraud patterns.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        chips(["Risk Scoring", "Explainable Cues", "Safe Next Steps", "Multilingual"], tone="accent"),
        unsafe_allow_html=True,
    )

    cta1, cta2 = st.columns(2)
    with cta1:
        if st.button("Open Detector", type="primary", use_container_width=True, key="open_detector_btn"):
            st.switch_page(_detector_page_target())
    with cta2:
        if st.button("Try Sample Scam", type="primary", use_container_width=True, key="try_sample_btn"):
            st.session_state["prefill_message"] = SAMPLE_SCAM_MESSAGE
            st.switch_page(_detector_page_target())

with right:
    particle_bits = _map_particle_bits()
    link_bits = _map_link_bits()
    packet_bits = _map_packet_bits()
    alert_red_bits = _map_node_bits(ALERT_RED_POINTS, "map-node-alert-red", "map-ring-alert-red", node_base=3.38, ring_base=4.86)
    alert_orange_bits = _map_node_bits(
        ALERT_ORANGE_POINTS, "map-node-alert-orange", "map-ring-alert-orange", node_base=3.30, ring_base=4.76
    )
    defense_green_bits = _map_node_bits(
        DEFENSE_GREEN_POINTS, "map-node-defense-green", "map-ring-defense-green", node_base=2.94, ring_base=4.02
    )
    defense_cyan_bits = _map_node_bits(
        DEFENSE_CYAN_POINTS, "map-node-defense-cyan", "map-ring-defense-cyan", node_base=2.90, ring_base=3.96
    )
    map_markup = (
        f"<div class='india-map-card'>"
        f"<div class='india-map-wrap'>"
        f"<svg class='india-map-svg' viewBox='22 22 370 370' role='img' aria-label='Animated India scam risk map'>"
        f"<defs><clipPath id='india-map-clip' clipPathUnits='userSpaceOnUse'><path d='{INDIA_OUTLINE_PATH}' /></clipPath></defs>"
        f"<path class='india-outline-main' d='{INDIA_OUTLINE_PATH}' fill='rgba(0,255,159,0.10)' stroke='rgba(0,255,159,0.92)' stroke-width='2.2' />"
        f"<g clip-path='url(#india-map-clip)'>"
        f"{link_bits}"
        f"{packet_bits}"
        f"{particle_bits}"
        f"{alert_red_bits}"
        f"{alert_orange_bits}"
        f"{defense_green_bits}"
        f"{defense_cyan_bits}"
        f"</g>"
        f"</svg></div></div>"
    )
    st.markdown(
        map_markup,
        unsafe_allow_html=True,
    )

f1, f2, f3, f4 = st.columns(4, gap="small")
f1.markdown(
    """
    <div class="feature-item">
        <div class="feature-icon">🔒</div>
        <div class="feature-title">Privacy-first</div>
        <div class="feature-desc">no_personal_data_needed</div>
    </div>
    """,
    unsafe_allow_html=True,
)
f2.markdown(
    """
    <div class="feature-item">
        <div class="feature-icon">🌐</div>
        <div class="feature-title">Multilingual</div>
        <div class="feature-desc">real_message_style</div>
    </div>
    """,
    unsafe_allow_html=True,
)
f3.markdown(
    """
    <div class="feature-item">
        <div class="feature-icon">🔎</div>
        <div class="feature-title">Explainable</div>
        <div class="feature-desc">not_a_black_box</div>
    </div>
    """,
    unsafe_allow_html=True,
)
f4.markdown(
    """
    <div class="feature-item">
        <div class="feature-icon">🇮🇳</div>
        <div class="feature-title">India-aware</div>
        <div class="feature-desc">otp_kyc_upi_patterns</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="section-header">
        <span class="section-code">// 02</span>
        <div class="section-title">Why This Helps</div>
    </div>
    """,
    unsafe_allow_html=True,
)

w1, w2, w3 = st.columns(3, gap="small")
w1.markdown(
    """
    <div class="why-card">
        <div class="icon">🎯</div>
        <h3>Detect</h3>
        <p>Risk score + scam type classification in one view. Instant and context-aware.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
w2.markdown(
    """
    <div class="why-card">
        <div class="icon">🧠</div>
        <h3>Explain</h3>
        <p>Highlights links, urgency language and impersonation cues clearly.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
w3.markdown(
    """
    <div class="why-card">
        <div class="icon">✅</div>
        <h3>Act</h3>
        <p>Clear next steps: block, report and verify safely through official channels.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
