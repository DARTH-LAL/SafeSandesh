from __future__ import annotations

import html
import json
import os
from typing import Iterable

import streamlit as st
import streamlit.components.v1 as components


def _inject_grid_parallax_js() -> None:
    """Drive subtle grid drift/parallax with mouse movement."""
    components.html(
        """
        <script>
        (function () {
          try {
            const parentWin = window.parent;
            const doc = parentWin.document;
            const root = doc.documentElement;
            if (!root || root.dataset.gridParallaxBound === "1") return;
            root.dataset.gridParallaxBound = "1";

            let targetX = 0;
            let targetY = 0;
            let currentX = 0;
            let currentY = 0;
            let rafId = null;

            function tick() {
              currentX += (targetX - currentX) * 0.11;
              currentY += (targetY - currentY) * 0.11;

              root.style.setProperty("--grid-drift-x", currentX.toFixed(2) + "px");
              root.style.setProperty("--grid-drift-y", currentY.toFixed(2) + "px");
              root.style.setProperty("--glow-drift-x", (currentX * 0.95).toFixed(2) + "px");
              root.style.setProperty("--glow-drift-y", (currentY * 0.95).toFixed(2) + "px");

              const dx = Math.abs(targetX - currentX);
              const dy = Math.abs(targetY - currentY);
              if (dx > 0.05 || dy > 0.05) {
                rafId = parentWin.requestAnimationFrame(tick);
              } else {
                rafId = null;
              }
            }

            function ensureTick() {
              if (rafId === null) {
                rafId = parentWin.requestAnimationFrame(tick);
              }
            }

            function onMove(event) {
              const w = Math.max(parentWin.innerWidth || 1, 1);
              const h = Math.max(parentWin.innerHeight || 1, 1);
              const nx = event.clientX / w - 0.5;
              const ny = event.clientY / h - 0.5;
              targetX = nx * 44;
              targetY = ny * 32;
              ensureTick();
            }

            function onLeave() {
              targetX = 0;
              targetY = 0;
              ensureTick();
            }

            parentWin.addEventListener("mousemove", onMove, { passive: true });
            parentWin.addEventListener("mouseleave", onLeave, { passive: true });
          } catch (err) {}
        })();
        </script>
        """,
        height=0,
        width=0,
    )


def _inject_terminal_typewriter_js() -> None:
    """Run rotating char-by-char terminal activity feed in the status strip."""
    components.html(
        """
        <script>
        (function () {
          try {
            const parentWin = window.parent;
            const doc = parentWin.document;
            const reduceMotion =
              parentWin.matchMedia &&
              parentWin.matchMedia("(prefers-reduced-motion: reduce)").matches;
            if (parentWin.__scamTerminalFxObserver) {
              try { parentWin.__scamTerminalFxObserver.disconnect(); } catch (e) {}
              parentWin.__scamTerminalFxObserver = null;
            }
            if (parentWin.__scamTerminalFxInterval) {
              try { parentWin.clearInterval(parentWin.__scamTerminalFxInterval); } catch (e) {}
              parentWin.__scamTerminalFxInterval = null;
            }

            const bindTypewriter = (el) => {
              let lines = [];
              try {
                lines = JSON.parse(el.getAttribute("data-lines") || "[]");
              } catch (e) {}
              lines = (Array.isArray(lines) ? lines : []).filter((x) => typeof x === "string" && x.length > 0);
              if (!lines.length) return;

              if (reduceMotion) {
                el.textContent = lines[0];
                el.setAttribute("data-typing-running", "1");
                return;
              }

              const signature = lines.join("\\n");
              if (el.getAttribute("data-lines-sig") === signature && el.getAttribute("data-typing-running") === "1") {
                return;
              }

              if (el._typingTimer) {
                parentWin.clearTimeout(el._typingTimer);
              }
              el.setAttribute("data-lines-sig", signature);
              el.setAttribute("data-typing-running", "1");

              let lineIndex = 0;
              const startLine = () => {
                const full = lines[lineIndex % lines.length];
                el.textContent = "";
                let i = 0;
                const baseSpeed = Math.max(14, Math.min(36, Math.floor(1700 / Math.max(full.length, 1))));

                const typeNext = () => {
                  i += 1;
                  el.textContent = full.slice(0, i);
                  if (i < full.length) {
                    const jitter = Math.floor(Math.random() * 11);
                    el._typingTimer = parentWin.setTimeout(typeNext, baseSpeed + jitter);
                  } else {
                    const holdMs = 260 + Math.min(180, full.length * 3);
                    el._typingTimer = parentWin.setTimeout(() => {
                      lineIndex += 1;
                      startLine();
                    }, holdMs);
                  }
                };
                typeNext();
              };
              startLine();
            };

            const bindStatus = (el) => {
              let states = [];
              try {
                states = JSON.parse(el.getAttribute("data-states") || "[]");
              } catch (e) {}
              states = (Array.isArray(states) ? states : []).filter((x) => typeof x === "string" && x.length > 0);
              if (!states.length) return;

              const label = el.querySelector(".active-label");
              if (!label) return;

              const statusSig = states.join("\\n");
              if (el.getAttribute("data-status-sig") === statusSig && el.getAttribute("data-status-running") === "1") {
                return;
              }

              if (el._statusTimer) {
                parentWin.clearInterval(el._statusTimer);
              }

              el.setAttribute("data-status-sig", statusSig);
              el.setAttribute("data-status-running", "1");

              let idx = 0;
              const animateSwitch = !reduceMotion;
              const setStatus = (next, animate) => {
                idx = ((next % states.length) + states.length) % states.length;
                if (animate && animateSwitch) {
                  el.classList.add("is-switching");
                  parentWin.setTimeout(() => {
                    label.textContent = states[idx];
                    el.classList.remove("is-switching");
                  }, 110);
                } else {
                  label.textContent = states[idx];
                }
              };

              setStatus(0, false);
              el._statusTimer = parentWin.setInterval(() => {
                setStatus(idx + 1, true);
              }, 2800);
            };

            const bindAll = () => {
              doc.querySelectorAll(".terminal-typing[data-lines]").forEach(bindTypewriter);
              doc.querySelectorAll(".terminal-active[data-states]").forEach(bindStatus);
            };

            parentWin.__scamTerminalFxBindAll = bindAll;
            bindAll();

            const observer = new parentWin.MutationObserver(() => {
              bindAll();
            });
            observer.observe(doc.body, { childList: true, subtree: true });
            parentWin.__scamTerminalFxObserver = observer;

            const tick = parentWin.setInterval(bindAll, 900);
            parentWin.__scamTerminalFxInterval = tick;
          } catch (err) {}
        })();
        </script>
        """,
        height=0,
        width=0,
    )


def _build_terminal_activity_lines(status_line: str) -> list[str]:
    base = " ".join((status_line or "").split()).strip() or "engine online... awaiting input"
    extra = [
        "loading heuristic ruleset... otp_kyc_upi signatures armed",
        "syncing shortlink indicators... 42 feeds normalized",
        "calibrating risk thresholds... ece_watchdog nominal",
        "watching message stream... anomaly detector listening",
        "explainability graph ready... cue extraction active",
    ]

    lines: list[str] = [base]
    for item in extra:
        if item.lower() != base.lower():
            lines.append(item)
    return lines


def _inject_scroll_reveal_js() -> None:
    """Reveal sections/cards on scroll with staggered delays."""
    components.html(
        """
        <script>
        (function () {
          try {
            const parentWin = window.parent;
            const doc = parentWin.document;
            const reduceMotion =
              parentWin.matchMedia &&
              parentWin.matchMedia("(prefers-reduced-motion: reduce)").matches;

            if (parentWin.__scamRevealObserver) {
              try { parentWin.__scamRevealObserver.disconnect(); } catch (e) {}
              parentWin.__scamRevealObserver = null;
            }
            if (parentWin.__scamRevealInterval) {
              try { parentWin.clearInterval(parentWin.__scamRevealInterval); } catch (e) {}
              parentWin.__scamRevealInterval = null;
            }

            const selectors = [
              ".hero-panel",
              ".terminal-card",
              ".feature-item",
              ".section-header",
              ".why-card",
              ".panel",
              ".tier-card"
            ].join(",");

            const markTargets = (observer) => {
              const nodes = doc.querySelectorAll(selectors);
              let idx = 0;
              nodes.forEach((el) => {
                if (el.dataset.revealBound !== "1") {
                  el.dataset.revealBound = "1";
                  el.classList.add("reveal-item");
                  const delay = Math.min(520, (idx % 8) * 85);
                  el.style.setProperty("--reveal-delay", delay + "ms");
                }

                if (reduceMotion) {
                  el.classList.add("is-visible");
                } else if (!el.classList.contains("is-visible")) {
                  observer.observe(el);
                }
                idx += 1;
              });
            };

            const observer = new parentWin.IntersectionObserver(
              (entries) => {
                entries.forEach((entry) => {
                  if (entry.isIntersecting) {
                    entry.target.classList.add("is-visible");
                    observer.unobserve(entry.target);
                  }
                });
              },
              { threshold: 0.16, rootMargin: "0px 0px -8% 0px" }
            );

            markTargets(observer);

            const mutationObserver = new parentWin.MutationObserver(() => {
              markTargets(observer);
            });
            mutationObserver.observe(doc.body, { childList: true, subtree: true });
            parentWin.__scamRevealObserver = mutationObserver;

            const tick = parentWin.setInterval(() => {
              markTargets(observer);
            }, 1000);
            parentWin.__scamRevealInterval = tick;
          } catch (err) {}
        })();
        </script>
        """,
        height=0,
        width=0,
    )


def _inject_data_stream_particles_js(enabled: bool) -> None:
    """Render lightweight neon data-stream particles in the background."""
    enabled_js = "true" if enabled else "false"
    components.html(
        f"""
        <script>
        (function () {{
          try {{
            const enabled = {enabled_js};
            const parentWin = window.parent;
            const doc = parentWin.document;
            const reduceMotion =
              parentWin.matchMedia &&
              parentWin.matchMedia("(prefers-reduced-motion: reduce)").matches;
            const stateKey = "__scamDataStreamFx";

            const cleanup = () => {{
              const state = parentWin[stateKey];
              if (!state) return;
              if (state.raf) {{
                try {{ parentWin.cancelAnimationFrame(state.raf); }} catch (e) {{}}
              }}
              if (state.onResize) {{
                try {{ parentWin.removeEventListener("resize", state.onResize); }} catch (e) {{}}
              }}
              if (state.canvas && state.canvas.parentNode) {{
                try {{ state.canvas.parentNode.removeChild(state.canvas); }} catch (e) {{}}
              }}
              parentWin[stateKey] = null;
            }};

            if (!enabled || reduceMotion) {{
              cleanup();
              return;
            }}

            const host = doc.querySelector(".stApp");
            if (!host) return;

            cleanup();

            const canvas = doc.createElement("canvas");
            canvas.id = "scam-data-stream-canvas";
            canvas.setAttribute("aria-hidden", "true");
            canvas.style.position = "fixed";
            canvas.style.inset = "0";
            canvas.style.width = "100vw";
            canvas.style.height = "100vh";
            canvas.style.pointerEvents = "none";
            canvas.style.zIndex = "1";
            canvas.style.opacity = "0.78";
            canvas.style.mixBlendMode = "normal";
            host.appendChild(canvas);

            const ctx = canvas.getContext("2d", {{ alpha: true }});
            if (!ctx) return;

            let width = 0;
            let height = 0;
            let dpr = 1;
            let raf = 0;
            let frameCount = 0;
            const waves = [];
            const haze = [];
            const cloud = [];

            const buildWaves = () => {{
              waves.length = 0;
              const waveCount = width < 760 ? 5 : (width < 1200 ? 7 : 8);
              for (let i = 0; i < waveCount; i += 1) {{
                const depth = waveCount === 1 ? 1 : i / (waveCount - 1);
                waves.push({{
                  depth: depth,
                  baseY: height * (0.30 + (depth * 0.50)) + (Math.random() * 18 - 9),
                  amp: 14 + (depth * 30) + (Math.random() * 8),
                  freq: 0.0063 + (depth * 0.0033) + (Math.random() * 0.0015),
                  speed: 0.014 + (depth * 0.028) + (Math.random() * 0.005),
                  phase: Math.random() * Math.PI * 2,
                  spread: 0.7 + (depth * 2.1),
                  hue: (i % 3 === 0) ? "cyan" : "neon",
                  topBand: false,
                  alphaBoost: 1,
                }});
              }}

              const topWaveStart = waves.length;
              const topWaveCount = width < 760 ? 3 : 4;
              for (let i = 0; i < topWaveCount; i += 1) {{
                const tDepth = topWaveCount === 1 ? 0.2 : i / (topWaveCount - 1);
                waves.push({{
                  depth: 0.08 + tDepth * 0.26,
                  baseY: height * (0.10 + (tDepth * 0.18)) + (Math.random() * 22 - 11),
                  amp: 8 + (tDepth * 14) + (Math.random() * 5),
                  freq: 0.007 + (tDepth * 0.0022) + (Math.random() * 0.0012),
                  speed: 0.018 + (tDepth * 0.018) + (Math.random() * 0.004),
                  phase: Math.random() * Math.PI * 2,
                  spread: 0.45 + (tDepth * 1.2),
                  hue: "cyan",
                  topBand: true,
                  alphaBoost: 1.34,
                }});
              }}

              cloud.length = 0;
              const cloudCount = width < 760 ? 1200 : (width < 1200 ? 2100 : 3000);
              const topCloudExtra = width < 760 ? 360 : (width < 1200 ? 620 : 880);
              const totalCloud = cloudCount + topCloudExtra;
              for (let i = 0; i < totalCloud; i += 1) {{
                let wIdx = 0;
                if (i < topCloudExtra) {{
                  wIdx = topWaveStart + Math.floor(Math.random() * topWaveCount);
                }} else {{
                  wIdx = Math.floor(Math.random() * waves.length);
                }}
                const w = waves[wIdx];
                cloud.push({{
                  waveIdx: wIdx,
                  x: Math.random() * width,
                  offsetY: (Math.random() - 0.5) * (w.topBand ? (12 + (w.depth * 26)) : (18 + (w.depth * 42))),
                  drift: (Math.random() - 0.5) * (w.topBand ? (0.11 + w.depth * 0.2) : (0.16 + w.depth * 0.38)),
                  seed: Math.random() * Math.PI * 2,
                  alpha: (0.05 + (w.depth * 0.14) + (Math.random() * 0.08)) * (w.alphaBoost || 1),
                  size: w.topBand
                    ? 0.8 + Math.random() * (1.1 + w.depth * 1.5)
                    : 0.7 + Math.random() * (1.0 + w.depth * 2.0),
                  hue: Math.random() > 0.64 ? "cyan" : "neon",
                }});
              }}

              haze.length = 0;
              const hazeCount = width < 760 ? 160 : 300;
              for (let i = 0; i < hazeCount; i += 1) {{
                haze.push({{
                  x: Math.random() * width,
                  y: (height * 0.08) + Math.random() * (height * 0.72),
                  vx: 0.02 + Math.random() * 0.1,
                  vy: (Math.random() - 0.5) * 0.045,
                  a: 0.08 + Math.random() * 0.23,
                  s: 0.8 + Math.random() * 2.0,
                  hue: Math.random() > 0.75 ? "cyan" : "neon",
                }});
              }}
            }};

            const resize = () => {{
              dpr = Math.min(2, parentWin.devicePixelRatio || 1);
              width = Math.max(1, parentWin.innerWidth || 1);
              height = Math.max(1, parentWin.innerHeight || 1);
              canvas.width = Math.floor(width * dpr);
              canvas.height = Math.floor(height * dpr);
              canvas.style.width = width + "px";
              canvas.style.height = height + "px";
              ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
              buildWaves();
            }};

            const render = () => {{
              frameCount += 1;
              if ((frameCount % 2) !== 0) {{
                raf = parentWin.requestAnimationFrame(render);
                return;
              }}

              ctx.clearRect(0, 0, width, height);
              for (let i = 0; i < cloud.length; i += 1) {{
                const p = cloud[i];
                const wave = waves[p.waveIdx];
                const time = frameCount * wave.speed;

                p.x += p.drift + Math.sin((frameCount * 0.012) + p.seed) * 0.02;
                if (p.x < -16) p.x = width + 8;
                if (p.x > width + 16) p.x = -8;

                const carrier = Math.sin((p.x * wave.freq) + time + wave.phase);
                const modulation = Math.cos((p.x * wave.freq * 0.61) - (time * 1.24) + (wave.phase * 0.66));
                const micro = Math.sin((frameCount * 0.033) + (i * 0.013) + p.seed) * wave.spread;
                const y = wave.baseY + (carrier * wave.amp) + (modulation * wave.amp * 0.42) + p.offsetY + micro;

                if (y < -12 || y > height + 12) continue;

                const crest = 0.55 + (carrier * 0.45);
                const twinkle = 0.68 + 0.32 * Math.sin((frameCount * 0.09) + p.seed + i * 0.005);
                const alpha = p.alpha * (0.5 + crest * 0.95) * twinkle * (wave.alphaBoost || 1);
                const color = p.hue === "cyan"
                  ? "rgba(64,198,255," + alpha.toFixed(3) + ")"
                  : "rgba(0,255,200," + alpha.toFixed(3) + ")";

                ctx.fillStyle = color;
                ctx.fillRect(p.x, y, p.size, p.size);

                if ((wave.depth > 0.55 || wave.topBand) && (i % 19 === 0)) {{
                  const glow = p.hue === "cyan"
                    ? "rgba(64,198,255," + (alpha * 0.45).toFixed(3) + ")"
                    : "rgba(0,255,200," + (alpha * 0.45).toFixed(3) + ")";
                  ctx.fillStyle = glow;
                  ctx.fillRect(p.x - 0.9, y - 0.9, p.size + 1.8, p.size + 1.8);
                }}
              }}

              for (let i = 0; i < haze.length; i += 1) {{
                const h = haze[i];
                h.x += h.vx;
                h.y += h.vy;
                if (h.x > width + 6) h.x = -4;
                if (h.y < height * 0.12 || h.y > height * 0.86) h.vy *= -1;
                const pulse = 0.65 + 0.35 * Math.sin((frameCount * 0.05) + i * 0.19);
                const a = h.a * pulse * 0.62;
                const c = h.hue === "cyan"
                  ? "rgba(64,198,255," + a.toFixed(3) + ")"
                  : "rgba(0,255,200," + a.toFixed(3) + ")";
                ctx.fillStyle = c;
                ctx.fillRect(h.x, h.y, h.s, h.s);
              }}

              raf = parentWin.requestAnimationFrame(render);
            }};

            resize();
            const onResize = () => resize();
            parentWin.addEventListener("resize", onResize, {{ passive: true }});
            raf = parentWin.requestAnimationFrame(render);

            parentWin[stateKey] = {{
              canvas: canvas,
              raf: raf,
              onResize: onResize,
            }};
          }} catch (err) {{}}
        }})();
        </script>
        """,
        height=0,
        width=0,
    )


def _inject_map_burst_js(enabled: bool) -> None:
    """Trigger simulated burst spikes on random map regions every 8-15s."""
    if not enabled:
        return

    components.html(
        """
        <script>
        (function () {
          try {
            const parentWin = window.parent;
            const doc = parentWin.document;
            const reduceMotion =
              parentWin.matchMedia &&
              parentWin.matchMedia("(prefers-reduced-motion: reduce)").matches;

            if (parentWin.__scamMapBurstTimer) {
              try { parentWin.clearTimeout(parentWin.__scamMapBurstTimer); } catch (e) {}
              parentWin.__scamMapBurstTimer = null;
            }

            const findNodes = () =>
              Array.from(doc.querySelectorAll(".india-map-svg .map-burst-node[data-burst-key]"));

            const clearActive = () => {
              doc
                .querySelectorAll(".india-map-svg .map-burst-node.burst-active, .india-map-svg .map-burst-ring.burst-active")
                .forEach((el) => el.classList.remove("burst-active"));
            };

            const triggerBurst = () => {
              if (reduceMotion) return;
              const nodes = findNodes();
              if (!nodes.length) return;

              const burstCount = Math.random() < 0.58 ? 1 : 2;
              const selected = [];
              while (selected.length < burstCount && selected.length < nodes.length) {
                const candidate = nodes[Math.floor(Math.random() * nodes.length)];
                if (!selected.includes(candidate)) selected.push(candidate);
              }

              selected.forEach((node, idx) => {
                const key = node.getAttribute("data-burst-key");
                const startDelay = idx * 160;
                parentWin.setTimeout(() => {
                  node.classList.add("burst-active");
                  doc
                    .querySelectorAll(".india-map-svg .map-burst-ring[data-burst-key='" + key + "']")
                    .forEach((ring) => ring.classList.add("burst-active"));
                  parentWin.setTimeout(() => {
                    node.classList.remove("burst-active");
                    doc
                      .querySelectorAll(".india-map-svg .map-burst-ring[data-burst-key='" + key + "']")
                      .forEach((ring) => ring.classList.remove("burst-active"));
                  }, 1500);
                }, startDelay);
              });
            };

            const schedule = () => {
              const delay = 8000 + Math.floor(Math.random() * 7000);
              parentWin.__scamMapBurstTimer = parentWin.setTimeout(() => {
                triggerBurst();
                schedule();
              }, delay);
            };

            clearActive();
            if (!reduceMotion) {
              parentWin.setTimeout(triggerBurst, 1800);
              schedule();
            }
          } catch (err) {}
        })();
        </script>
        """,
        height=0,
        width=0,
    )


def apply_theme(home_particles: bool = False) -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Oxanium:wght@300;400;600;700;800&display=swap');

        :root {
            --neon: #00ff9f;
            --neon2: #00d4ff;
            --red: #ff3860;
            --yellow: #ffdd57;
            --bg: #020810;
            --surface: rgba(3, 15, 26, 0.90);
            --surface-2: rgba(5, 21, 37, 0.92);
            --border: rgba(0,255,159,0.16);
            --text: #c8f0e0;
            --muted: rgba(200,240,224,0.62);
            --nav-inactive: #ff5a66;
            --scanline-a: rgba(0,255,159,0.00);
            --scanline-b: rgba(0,255,159,0.22);
            --scanline-c: rgba(0,212,255,0.34);
            --grid-drift-x: 0px;
            --grid-drift-y: 0px;
            --glow-drift-x: 0px;
            --glow-drift-y: 0px;
        }

        html, body {
            margin: 0;
            padding: 0;
            background: var(--bg);
        }

        .stApp {
            color: var(--text);
            font-family: 'Oxanium', sans-serif;
            background: var(--bg);
            position: relative;
            min-height: 100vh;
            overflow-x: hidden;
            isolation: isolate;
        }

        .stApp::before {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            z-index: 0;
            background-image:
              linear-gradient(rgba(0,255,159,0.075) 1px, transparent 1px),
              linear-gradient(90deg, rgba(0,255,159,0.075) 1px, transparent 1px);
            background-size: 40px 40px, 40px 40px;
            transform: translate3d(var(--grid-drift-x), var(--grid-drift-y), 0);
            transition: transform 90ms linear;
            will-change: transform;
            filter: drop-shadow(0 0 2px rgba(0,255,159,0.22));
        }

        .stApp::after {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            z-index: 0;
            background:
              radial-gradient(ellipse 45% 35% at 15% 20%, rgba(0,255,159,0.16) 0%, transparent 70%),
              radial-gradient(ellipse 40% 35% at 85% 70%, rgba(0,212,255,0.14) 0%, transparent 70%);
            transform: translate3d(var(--glow-drift-x), var(--glow-drift-y), 0);
            transition: transform 110ms linear;
            will-change: transform;
        }

        [data-testid="stAppViewContainer"],
        [data-testid="stAppViewContainer"] > .main {
            position: relative;
            z-index: 2;
            background: transparent !important;
        }

        /* Disable Streamlit sidebar navigation; use custom top nav only */
        [data-testid="stSidebar"],
        [data-testid="stSidebarNav"],
        [data-testid="collapsedControl"] {
            display: none !important;
        }

        .main .block-container {
            max-width: 1180px;
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }

        h1, h2, h3, h4 {
            color: var(--text);
            letter-spacing: 0.02em;
        }

        .stMarkdown, .stCaption {
            color: var(--muted);
        }

        .reveal-item {
            opacity: 0;
            transform: translateY(14px) scale(0.995);
            filter: saturate(0.92);
            transition:
                opacity 420ms cubic-bezier(0.18, 0.82, 0.24, 1),
                transform 420ms cubic-bezier(0.18, 0.82, 0.24, 1),
                filter 420ms ease;
            transition-delay: var(--reveal-delay, 0ms);
            will-change: transform, opacity;
        }

        .reveal-item.is-visible {
            opacity: 1;
            transform: translateY(0) scale(1);
            filter: none;
        }

        .menu-wrap {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.8rem;
            background: rgba(2, 8, 16, 0.90);
            border: 1px solid var(--border);
            padding: 0.7rem 1rem;
            margin-bottom: 0.55rem;
        }

        .menu-logo {
            font-family: 'Share Tech Mono', monospace;
            color: var(--neon);
            letter-spacing: 0.08em;
            text-shadow: 0 0 10px rgba(0,255,159,0.45);
            font-size: 1.04rem;
            white-space: nowrap;
        }

        .menu-links {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            flex-wrap: wrap;
            justify-content: flex-end;
        }

        .menu-link {
            text-decoration: none;
            color: var(--nav-inactive);
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.78rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            padding: 0.46rem 0.85rem;
            border: 1px solid transparent;
            transition: color 180ms ease, background-color 180ms ease, border-color 180ms ease;
            position: relative;
            z-index: 2;
            text-shadow: 0 0 8px rgba(255, 90, 102, 0.18);
        }

        .menu-links .menu-link,
        .menu-links .menu-link:link,
        .menu-links .menu-link:visited,
        .menu-links .menu-link:focus,
        .menu-links .menu-link:active {
            color: var(--nav-inactive) !important;
            -webkit-text-fill-color: var(--nav-inactive) !important;
            text-decoration: none !important;
        }

        .menu-links .menu-link:hover {
            color: var(--neon) !important;
            -webkit-text-fill-color: var(--neon) !important;
            border-color: var(--border);
            background: rgba(0,255,159,0.04);
            text-shadow: 0 0 10px rgba(0,255,159,0.34);
        }

        .menu-links .menu-link.active,
        .menu-links .menu-link.active:link,
        .menu-links .menu-link.active:visited,
        .menu-links .menu-link.active:focus,
        .menu-links .menu-link.active:active {
            color: var(--neon) !important;
            -webkit-text-fill-color: var(--neon) !important;
            border-color: transparent;
            background: rgba(0,255,159,0.04);
            box-shadow: none;
            text-shadow: 0 0 10px rgba(0,255,159,0.45);
        }

        .terminal-strip {
            display: flex;
            align-items: center;
            gap: 0.9rem;
            flex-wrap: wrap;
            border: 1px solid var(--border);
            background: rgba(3, 15, 26, 0.84);
            padding: 0.6rem 1rem;
            margin-bottom: 1.1rem;
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.72rem;
            letter-spacing: 0.08em;
        }

        .terminal-typing-wrap {
            display: inline-flex;
            align-items: center;
            min-width: 0;
            flex: 1 1 360px;
            max-width: 760px;
        }

        .terminal-typing {
            display: inline-block;
            white-space: nowrap;
            overflow: hidden;
            max-width: 100%;
            color: var(--muted);
            position: relative;
            padding-right: 0.2rem;
        }

        .terminal-typing::after {
            content: "";
            display: inline-block;
            width: 0.58ch;
            height: 1.02em;
            margin-left: 0.12ch;
            vertical-align: -0.12em;
            background: rgba(0,255,159,0.82);
            box-shadow: 0 0 8px rgba(0,255,159,0.6);
            animation: terminal-caret 0.85s steps(1, end) infinite;
        }

        @keyframes terminal-caret {
            0%, 46% { opacity: 1; }
            47%, 100% { opacity: 0.12; }
        }

        .terminal-prompt {
            color: var(--neon);
            font-weight: 700;
        }

        .terminal-active {
            color: var(--neon);
            display: inline-flex;
            align-items: center;
            gap: 0;
            margin-left: auto;
            padding: 0.15rem 0.42rem;
            border: 1px solid rgba(0,255,159,0.2);
            background: rgba(0,255,159,0.05);
            white-space: nowrap;
            min-width: 13ch;
            justify-content: flex-start;
            font-weight: 700;
        }

        .terminal-active .active-label {
            display: inline-block;
            min-width: 11ch;
            transition: opacity 0.16s ease, transform 0.16s ease;
        }

        .terminal-active.is-switching .active-label {
            opacity: 0.05;
            transform: translateY(-2px);
        }

        .hero-panel,
        .panel,
        .terminal-card,
        .feature-item,
        .why-card,
        .tier-card {
            background: var(--surface);
            border: 1px solid var(--border);
            position: relative;
            overflow: hidden;
            isolation: isolate;
        }

        .hero-panel::after,
        .panel::after,
        .terminal-card::after,
        .feature-item::after,
        .why-card::after,
        .tier-card::after {
            content: "";
            position: absolute;
            top: -120%;
            left: -150%;
            width: 90%;
            height: 300%;
            background: linear-gradient(
                112deg,
                var(--scanline-a) 0%,
                var(--scanline-a) 40%,
                var(--scanline-b) 49%,
                var(--scanline-c) 50%,
                var(--scanline-b) 51%,
                var(--scanline-a) 60%,
                var(--scanline-a) 100%
            );
            opacity: 0.28;
            pointer-events: none;
            mix-blend-mode: screen;
            transform: translateX(-120%);
            animation: neon-scan-sweep 9s linear infinite;
            will-change: transform;
        }

        .hero-panel:hover::after,
        .panel:hover::after,
        .terminal-card:hover::after,
        .feature-item:hover::after,
        .why-card:hover::after,
        .tier-card:hover::after {
            opacity: 0.42;
            animation-duration: 6.5s;
        }

        .terminal-card:not(.india-map-card),
        .feature-item,
        .why-card {
            transform: perspective(1000px) rotateX(0deg) rotateY(0deg) translateY(0);
            transform-style: preserve-3d;
            transition:
                transform 200ms ease,
                box-shadow 240ms ease,
                border-color 240ms ease,
                filter 240ms ease;
            will-change: transform, box-shadow;
        }

        .terminal-card:not(.india-map-card):hover,
        .feature-item:hover,
        .why-card:hover {
            transform: perspective(1000px) rotateX(1.15deg) rotateY(-1.35deg) translateY(-2px);
            border-color: rgba(0,255,159,0.42);
            box-shadow:
                0 10px 20px rgba(0,0,0,0.34),
                0 0 0 1px rgba(0,255,159,0.24),
                0 0 22px rgba(0,255,159,0.22);
            filter: saturate(1.08);
        }

        @keyframes neon-scan-sweep {
            0% {
                transform: translateX(-120%);
            }
            100% {
                transform: translateX(220%);
            }
        }

        .hero-panel {
            padding: 1.05rem;
            margin-bottom: 0.95rem;
            box-shadow: 0 0 24px rgba(0,255,159,0.06);
        }

        .hero-kicker {
            font-family: 'Share Tech Mono', monospace;
            color: var(--neon2);
            font-size: 0.72rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 0.32rem;
        }

        .hero-title {
            margin: 0;
            color: var(--text);
            font-size: 1.5rem;
            font-weight: 700;
            text-transform: uppercase;
        }

        .hero-sub2 {
            margin: 0.42rem 0 0 0;
            color: var(--muted);
            font-size: 0.95rem;
            line-height: 1.5;
        }

        .status-bar {
            display: flex;
            align-items: center;
            gap: 0.8rem;
            margin-bottom: 0.7rem;
        }

        .status-dot {
            width: 9px;
            height: 9px;
            border-radius: 50%;
            background: var(--neon);
            box-shadow:
                0 0 10px rgba(0,255,159,0.85),
                0 0 20px rgba(0,255,159,0.4);
            position: relative;
            isolation: isolate;
            animation: system-dot-core 1.9s ease-in-out infinite;
        }

        .status-dot::before,
        .status-dot::after {
            content: "";
            position: absolute;
            left: 50%;
            top: 50%;
            width: 100%;
            height: 100%;
            border-radius: 50%;
            border: 1px solid rgba(0,255,159,0.42);
            transform: translate(-50%, -50%) scale(1);
            opacity: 0;
            pointer-events: none;
            animation: system-dot-ring 2.2s ease-out infinite;
        }

        .status-dot::after {
            animation-delay: 1.1s;
            border-color: rgba(0,212,255,0.38);
        }

        @keyframes system-dot-core {
            0%, 100% {
                transform: scale(1);
                box-shadow:
                    0 0 10px rgba(0,255,159,0.85),
                    0 0 20px rgba(0,255,159,0.4);
            }
            50% {
                transform: scale(1.22);
                box-shadow:
                    0 0 14px rgba(0,255,159,1),
                    0 0 30px rgba(0,255,159,0.56);
            }
        }

        @keyframes system-dot-ring {
            0% {
                transform: translate(-50%, -50%) scale(1);
                opacity: 0.65;
            }
            80% {
                transform: translate(-50%, -50%) scale(3.3);
                opacity: 0.08;
            }
            100% {
                transform: translate(-50%, -50%) scale(3.55);
                opacity: 0;
            }
        }

        .status-text {
            font-family: 'Share Tech Mono', monospace;
            color: var(--neon);
            font-size: 0.72rem;
            letter-spacing: 0.1em;
            text-transform: uppercase;
        }

        .version-tag {
            font-family: 'Share Tech Mono', monospace;
            color: var(--muted);
            border: 1px solid var(--border);
            padding: 0.12rem 0.45rem;
            font-size: 0.68rem;
            letter-spacing: 0.08em;
        }

        .cyber-title {
            margin: 0;
            font-size: clamp(2.2rem, 4vw, 3.4rem);
            line-height: 1.05;
            font-weight: 800;
        }

        .cyber-title .line1 {
            display: block;
            color: var(--text);
        }

        .cyber-title .line2 {
            display: block;
            background: linear-gradient(90deg, var(--neon), var(--neon2));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .cyber-title .line3 {
            display: block;
            color: transparent;
            -webkit-text-stroke: 1px rgba(0,255,159,0.45);
            font-size: 0.84em;
        }

        .hero-sub {
            color: var(--muted);
            line-height: 1.7;
            margin: 0.55rem 0 1rem 0;
            max-width: 470px;
        }

        .terminal-card {
            padding: 0;
            box-shadow: 0 0 26px rgba(0,255,159,0.06);
        }

        .india-map-card {
            padding: 0;
            margin-bottom: 1rem;
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            position: relative;
            z-index: 1;
            transform: none !important;
            overflow: visible;
        }

        .india-map-card::after {
            display: none !important;
        }

        .india-map-wrap {
            position: relative;
            width: 100%;
            height: clamp(390px, 54vh, 620px);
            border: none;
            background:
                radial-gradient(78% 62% at 36% 42%, rgba(0,255,159,0.10) 0%, rgba(0,255,159,0.00) 72%),
                radial-gradient(62% 52% at 67% 38%, rgba(0,212,255,0.11) 0%, rgba(0,212,255,0.00) 70%);
            overflow: visible;
            isolation: isolate;
        }

        .india-map-wrap::after {
            display: none;
        }

        .india-map-svg {
            width: 100%;
            height: 100%;
            display: block;
            overflow: visible;
            transform: scale(1.05);
            transform-origin: center;
        }

        .india-outline-main,
        .india-outline-ne {
            stroke: rgba(0,255,159,0.92);
            stroke-width: 2.2;
            stroke-linejoin: round;
            stroke-linecap: round;
            stroke-dasharray: 7 8.5;
            filter: drop-shadow(0 0 10px rgba(0,255,159,0.42));
            animation: india-outline-dash 8.5s linear infinite;
        }

        .india-outline-ne {
            stroke-width: 1.95;
        }

        .india-link {
            fill: none;
            stroke: rgba(0,212,255,0.35);
            stroke-width: 1.1;
            stroke-dasharray: 3 6;
            animation: india-link-flow 5.4s linear infinite;
        }

        .map-packet {
            opacity: 0.24;
            pointer-events: none;
            mix-blend-mode: screen;
            animation: map-packet-flicker 1.26s ease-in-out infinite;
        }

        .map-packet-neon {
            fill: rgba(0,255,185,0.54);
            filter: drop-shadow(0 0 5px rgba(0,255,185,0.42));
        }

        .map-packet-cyan {
            fill: rgba(64,198,255,0.5);
            filter: drop-shadow(0 0 5px rgba(64,198,255,0.42));
        }

        .map-bit {
            animation: map-bit-twinkle 3.1s ease-in-out infinite;
        }

        .map-bit-neon {
            fill: rgba(0,255,200,0.78);
            filter: drop-shadow(0 0 5px rgba(0,255,200,0.45));
        }

        .map-bit-cyan {
            fill: rgba(64,198,255,0.78);
            filter: drop-shadow(0 0 5px rgba(64,198,255,0.45));
        }

        .map-node-alert-red {
            fill: rgba(255,66,92,0.97);
            filter: drop-shadow(0 0 12px rgba(255,66,92,0.88));
            animation: map-hotspot-glow-red 1.15s ease-in-out infinite;
        }

        .map-node-alert-orange {
            fill: rgba(255,160,54,0.97);
            filter: drop-shadow(0 0 12px rgba(255,160,54,0.84));
            animation: map-hotspot-glow-orange 1.28s ease-in-out infinite;
        }

        .map-node-defense-green {
            fill: rgba(0,255,159,0.98);
            filter: drop-shadow(0 0 11px rgba(0,255,159,0.82));
            animation: map-defense-glow-green 1.45s ease-in-out infinite;
        }

        .map-node-defense-cyan {
            fill: rgba(64,198,255,0.98);
            filter: drop-shadow(0 0 11px rgba(64,198,255,0.78));
            animation: map-defense-glow-cyan 1.62s ease-in-out infinite;
        }

        .map-ring-alert-red,
        .map-ring-alert-orange,
        .map-ring-defense-green,
        .map-ring-defense-cyan {
            fill: none;
            transform-box: fill-box;
            transform-origin: center;
            animation: map-ring-glow 2.05s ease-in-out infinite;
        }

        .map-ring-alert-red {
            stroke: rgba(255,66,92,0.8);
            stroke-width: 1.28;
            animation-duration: 1.45s;
            filter: drop-shadow(0 0 8px rgba(255,66,92,0.62));
        }

        .map-ring-alert-orange {
            stroke: rgba(255,160,54,0.76);
            stroke-width: 1.22;
            animation-duration: 1.62s;
            filter: drop-shadow(0 0 8px rgba(255,160,54,0.58));
        }

        .map-ring-defense-green {
            stroke: rgba(0,255,159,0.72);
            stroke-width: 1.18;
            animation-duration: 1.86s;
            filter: drop-shadow(0 0 7px rgba(0,255,159,0.56));
        }

        .map-ring-defense-cyan {
            stroke: rgba(64,198,255,0.74);
            stroke-width: 1.16;
            animation-duration: 2.05s;
            filter: drop-shadow(0 0 7px rgba(64,198,255,0.54));
        }

        .map-burst-node,
        .map-burst-ring {
            transition: opacity 170ms ease, filter 180ms ease, stroke-width 180ms ease;
        }

        .map-node-alert-red.burst-active {
            opacity: 1 !important;
            filter: drop-shadow(0 0 21px rgba(255,66,92,1)) !important;
        }

        .map-node-alert-orange.burst-active {
            opacity: 1 !important;
            filter: drop-shadow(0 0 20px rgba(255,160,54,0.98)) !important;
        }

        .map-node-defense-green.burst-active {
            opacity: 1 !important;
            filter: drop-shadow(0 0 18px rgba(0,255,159,1)) !important;
        }

        .map-node-defense-cyan.burst-active {
            opacity: 1 !important;
            filter: drop-shadow(0 0 18px rgba(64,198,255,0.96)) !important;
        }

        .map-ring-alert-red.burst-active,
        .map-ring-alert-orange.burst-active {
            opacity: 0.95 !important;
            stroke-width: 2.0 !important;
        }

        .map-ring-defense-green.burst-active,
        .map-ring-defense-cyan.burst-active {
            opacity: 0.9 !important;
            stroke-width: 1.8 !important;
        }

        @keyframes india-outline-dash {
            from { stroke-dashoffset: 0; }
            to { stroke-dashoffset: -260; }
        }

        @keyframes india-link-flow {
            from { stroke-dashoffset: 0; opacity: 0.3; }
            50% { opacity: 0.62; }
            to { stroke-dashoffset: -140; opacity: 0.3; }
        }

        @keyframes map-bit-twinkle {
            0%, 100% { opacity: 0.38; }
            50% { opacity: 1; }
        }

        @keyframes map-packet-flicker {
            0%, 100% { opacity: 0.18; }
            45% { opacity: 0.58; }
            55% { opacity: 0.36; }
        }

        @keyframes map-hotspot-glow-red {
            0%, 100% { opacity: 0.56; filter: drop-shadow(0 0 5px rgba(255,66,92,0.48)); }
            22% { opacity: 0.98; filter: drop-shadow(0 0 15px rgba(255,66,92,0.98)); }
            30% { opacity: 0.62; filter: drop-shadow(0 0 6px rgba(255,66,92,0.52)); }
            55% { opacity: 1; filter: drop-shadow(0 0 17px rgba(255,66,92,1)); }
            63% { opacity: 0.58; filter: drop-shadow(0 0 6px rgba(255,66,92,0.5)); }
        }

        @keyframes map-hotspot-glow-orange {
            0%, 100% { opacity: 0.54; filter: drop-shadow(0 0 5px rgba(255,160,54,0.46)); }
            24% { opacity: 0.95; filter: drop-shadow(0 0 14px rgba(255,160,54,0.94)); }
            34% { opacity: 0.6; filter: drop-shadow(0 0 6px rgba(255,160,54,0.5)); }
            58% { opacity: 1; filter: drop-shadow(0 0 16px rgba(255,160,54,0.98)); }
            66% { opacity: 0.56; filter: drop-shadow(0 0 6px rgba(255,160,54,0.48)); }
        }

        @keyframes map-defense-glow-green {
            0%, 100% { opacity: 0.58; filter: drop-shadow(0 0 5px rgba(0,255,159,0.42)); }
            26% { opacity: 0.96; filter: drop-shadow(0 0 13px rgba(0,255,159,0.9)); }
            36% { opacity: 0.64; filter: drop-shadow(0 0 6px rgba(0,255,159,0.5)); }
            60% { opacity: 1; filter: drop-shadow(0 0 15px rgba(0,255,159,0.95)); }
            68% { opacity: 0.6; filter: drop-shadow(0 0 6px rgba(0,255,159,0.46)); }
        }

        @keyframes map-defense-glow-cyan {
            0%, 100% { opacity: 0.56; filter: drop-shadow(0 0 5px rgba(64,198,255,0.42)); }
            24% { opacity: 0.94; filter: drop-shadow(0 0 12px rgba(64,198,255,0.86)); }
            34% { opacity: 0.62; filter: drop-shadow(0 0 6px rgba(64,198,255,0.48)); }
            58% { opacity: 0.99; filter: drop-shadow(0 0 14px rgba(64,198,255,0.9)); }
            66% { opacity: 0.58; filter: drop-shadow(0 0 6px rgba(64,198,255,0.46)); }
        }

        @keyframes map-ring-glow {
            0%, 100% { opacity: 0.18; stroke-width: 0.92; }
            28% { opacity: 0.84; stroke-width: 1.5; }
            36% { opacity: 0.26; stroke-width: 1.0; }
            62% { opacity: 0.88; stroke-width: 1.56; }
            70% { opacity: 0.22; stroke-width: 0.96; }
        }

        @keyframes india-map-sweep {
            0% { transform: translateX(-70%) rotate(-2deg); opacity: 0; }
            10% { opacity: 0.8; }
            65% { opacity: 0.45; }
            100% { transform: translateX(68%) rotate(-2deg); opacity: 0; }
        }

        .t-titlebar {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 0.55rem 0.9rem;
            border-bottom: 1px solid var(--border);
            background: rgba(0,0,0,0.22);
        }

        .t-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
        }

        .t-dot-red { background: var(--red); box-shadow: 0 0 7px var(--red); }
        .t-dot-yellow { background: var(--yellow); box-shadow: 0 0 7px var(--yellow); }
        .t-dot-green { background: var(--neon); box-shadow: 0 0 7px var(--neon); }

        .t-title {
            margin-left: auto;
            font-family: 'Share Tech Mono', monospace;
            color: var(--muted);
            font-size: 0.7rem;
            letter-spacing: 0.08em;
        }

        .t-body {
            padding: 0.9rem;
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.78rem;
            color: var(--muted);
            line-height: 1.7;
        }

        .t-line-prompt { color: rgba(0,255,159,0.5); }
        .t-line-input { color: var(--text); }
        .t-line-output { margin-left: 14px; }

        .t-divider {
            border: none;
            border-top: 1px solid var(--border);
            margin: 0.8rem 0;
        }

        .t-alert {
            border: 1px solid rgba(255,56,96,0.42);
            background: rgba(255,56,96,0.06);
            padding: 0.72rem;
        }

        .t-alert-header {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 0.5rem;
        }

        .alert-icon { color: var(--red); }

        .alert-title {
            color: var(--red);
            font-weight: 700;
            letter-spacing: 0.08em;
            font-size: 0.75rem;
        }

        .feature-item {
            padding: 0.9rem;
            height: 100%;
        }

        .feature-icon { font-size: 1.1rem; margin-bottom: 0.35rem; }

        .feature-title {
            color: var(--text);
            text-transform: uppercase;
            letter-spacing: 0.06em;
            font-size: 0.8rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .feature-desc {
            color: var(--muted);
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.68rem;
        }

        .section-header {
            display: flex;
            align-items: baseline;
            gap: 0.7rem;
            margin: 1.4rem 0 0.7rem;
        }

        .section-code {
            color: var(--neon);
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.74rem;
            opacity: 0.75;
        }

        .section-title {
            color: var(--text);
            font-size: 1.45rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            font-weight: 700;
        }

        .why-card {
            padding: 1rem;
            height: 100%;
        }

        .why-card .icon { font-size: 1.2rem; margin-bottom: 0.4rem; }

        .why-card h3 {
            margin: 0 0 0.3rem;
            color: var(--neon);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.95rem;
        }

        .why-card p {
            margin: 0;
            color: var(--muted);
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.74rem;
            line-height: 1.65;
        }

        .panel {
            padding: 0.88rem;
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.55;
        }

        .chip {
            display: inline-block;
            border: 1px solid rgba(0,212,255,0.24);
            color: var(--neon2);
            background: rgba(0,212,255,0.07);
            padding: 0.24rem 0.55rem;
            margin-right: 0.32rem;
            margin-bottom: 0.32rem;
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.7rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .chip-safe { border-color: rgba(0,255,159,0.3); color: var(--neon); background: rgba(0,255,159,0.08); }
        .chip-suspicious { border-color: rgba(255,221,87,0.35); color: var(--yellow); background: rgba(255,221,87,0.08); }
        .chip-phishing { border-color: rgba(255,56,96,0.40); color: var(--red); background: rgba(255,56,96,0.08); }
        .chip-accent { border-color: rgba(0,212,255,0.24); color: var(--neon2); background: rgba(0,212,255,0.07); }

        .risk-track {
            width: 100%;
            height: 5px;
            background: rgba(255,255,255,0.08);
            border: 1px solid var(--border);
            overflow: hidden;
            margin-top: 0.3rem;
            margin-bottom: 0.52rem;
        }

        .risk-fill {
            height: 100%;
        }

        .stButton > button {
            border-radius: 0 !important;
            border: 1px solid var(--border) !important;
            background: transparent !important;
            color: var(--text) !important;
            font-family: 'Oxanium', sans-serif !important;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-size: 0.8rem !important;
            padding: 0.58rem 0.9rem !important;
            position: relative;
            overflow: hidden;
            isolation: isolate;
        }

        .stButton > button::before {
            content: "";
            position: absolute;
            top: -1px;
            bottom: -1px;
            left: -140%;
            width: 62%;
            pointer-events: none;
            z-index: 1;
            opacity: 0;
            transform: skewX(-18deg);
            background: linear-gradient(
                90deg,
                rgba(255,255,255,0.00) 0%,
                rgba(255,255,255,0.18) 30%,
                rgba(255,255,255,0.66) 50%,
                rgba(0,255,195,0.44) 72%,
                rgba(0,255,195,0.00) 100%
            );
            filter: blur(0.2px);
        }

        .stButton > button:hover::before {
            opacity: 1;
            animation: neon-button-sweep 920ms cubic-bezier(0.22, 0.7, 0.28, 1) 1;
        }

        @keyframes neon-button-sweep {
            0% {
                left: -140%;
                opacity: 0;
            }
            16% {
                opacity: 0.95;
            }
            100% {
                left: 140%;
                opacity: 0;
            }
        }

        .stButton > button > * {
            position: relative;
            z-index: 2;
        }

        .stButton > button[kind="primary"] {
            background: var(--neon) !important;
            border-color: transparent !important;
            color: #02170f !important;
            font-weight: 700 !important;
        }

        .st-key-open_detector_btn .stButton > button,
        .st-key-open_detector_btn .stButton > button[kind="primary"] {
            background: var(--red) !important;
            border-color: transparent !important;
            color: #fff8fb !important;
            text-shadow: none !important;
            box-shadow: 0 0 14px rgba(255,56,96,0.35) !important;
        }

        .st-key-open_detector_btn .stButton > button:hover {
            background: #ff4f75 !important;
            box-shadow: 0 0 18px rgba(255,56,96,0.46) !important;
        }

        .st-key-try_sample_btn .stButton > button,
        .st-key-try_sample_btn .stButton > button[kind="primary"] {
            background: var(--neon) !important;
            border-color: transparent !important;
            color: #02170f !important;
            box-shadow: 0 0 14px rgba(0,255,159,0.32) !important;
        }

        .stTextArea textarea,
        .stTextInput input,
        .stNumberInput input,
        .stDateInput input,
        .stTimeInput input,
        .stSelectbox div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div {
            background: var(--surface-2) !important;
            color: var(--text) !important;
            border: 1px solid var(--border) !important;
            border-radius: 0 !important;
        }

        .stTabs [data-baseweb="tab"] {
            border: 1px solid var(--border);
            background: var(--surface);
            color: var(--muted);
            border-radius: 0;
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.72rem;
            letter-spacing: 0.09em;
            text-transform: uppercase;
        }

        .stTabs [aria-selected="true"] {
            color: var(--neon) !important;
            border-color: rgba(0,255,159,0.35) !important;
        }

        @media (max-width: 860px) {
            .menu-wrap,
            .terminal-strip {
                padding-left: 0.75rem;
                padding-right: 0.75rem;
            }

            .menu-logo {
                display: none;
            }

            .menu-links {
                width: 100%;
                justify-content: center;
            }

            .terminal-typing-wrap {
                flex: 1 1 100%;
                max-width: 100%;
            }

            .terminal-typing {
                max-width: 100%;
            }
        }

        @media (prefers-reduced-motion: reduce) {
            .reveal-item,
            .reveal-item.is-visible {
                opacity: 1 !important;
                transform: none !important;
                filter: none !important;
                transition: none !important;
            }
            .india-map-wrap::after,
            .india-outline-main,
            .india-outline-ne,
            .india-link,
            .map-packet,
            .map-bit,
            .map-node-alert-red,
            .map-node-alert-orange,
            .map-node-defense-green,
            .map-node-defense-cyan,
            .map-ring-alert-red,
            .map-ring-alert-orange,
            .map-ring-defense-green,
            .map-ring-defense-cyan {
                animation: none !important;
            }
            .terminal-typing {
                max-width: 100%;
            }
            .terminal-typing::after {
                animation: none !important;
                opacity: 0.45;
            }
            .stButton > button::before {
                animation: none !important;
                opacity: 0 !important;
            }
            .terminal-card,
            .feature-item,
            .why-card,
            .terminal-card:hover,
            .feature-item:hover,
            .why-card:hover {
                transform: none !important;
                filter: none !important;
            }
            .india-map-card {
                margin-bottom: 0.4rem;
            }
            .india-map-wrap {
                height: clamp(300px, 41vh, 440px);
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    _inject_grid_parallax_js()
    _inject_terminal_typewriter_js()
    _inject_scroll_reveal_js()
    _inject_data_stream_particles_js(home_particles)
    _inject_map_burst_js(home_particles)


def top_menu(current: str = "home") -> None:
    key = (current or "home").strip().lower()
    shell = os.environ.get("SAFESANDESH_APP_SHELL", "combined").strip().lower()
    if shell == "consumer":
        links = [
            ("Home", "/", "home"),
            ("Detector", "/Detector", "detector"),
            ("Dashboard", "/Dashboard", "dashboard"),
            ("Analyst Lab", "/Analyst_Lab", "analyst_lab"),
        ]
        if key in {"analyst_lab", "ai_studio"}:
            links.append(("AI Studio", "/AI_Studio", "ai_studio"))
    elif shell == "technical":
        links = [
            ("Dashboard", "/", "dashboard"),
            ("AI Studio", "/AI_Studio", "ai_studio"),
        ]
    else:
        # Default to the consumer navigation if someone runs app.py directly.
        # The technical side must be launched through apps/technical_app.py.
        links = [
            ("Home", "/", "home"),
            ("Detector", "/Detector", "detector"),
            ("Dashboard", "/Dashboard", "dashboard"),
        ]

    nav_links = []
    for label, href, id_key in links:
        active_match = key == id_key or (id_key == "analyst_lab" and key == "technical_dashboard")
        active = " active" if active_match else ""
        safe_href = html.escape(href, quote=True)
        nav_links.append(
            (
                f"<a class='menu-link{active}' href='{safe_href}' target='_self' "
                f"onclick=\"window.location.assign('{safe_href}'); return false;\">{html.escape(label)}</a>"
            )
        )

    st.markdown(
        (
            "<div class='menu-wrap'>"
            "<div class='menu-logo'>SafeSandesh</div>"
            f"<div class='menu-links'>{''.join(nav_links)}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def terminal_header(status_line: str) -> None:
    lines = _build_terminal_activity_lines(status_line)
    initial_status = html.escape(lines[0])
    lines_json = html.escape(json.dumps(lines), quote=True)
    status_states = ["SYSTEM ACTIVE", "SCANNING", "SYNCED"]
    states_json = html.escape(json.dumps(status_states), quote=True)
    st.markdown(
        (
            "<div class='terminal-strip'>"
            "<span class='terminal-prompt'>root@safesandesh:~$</span>"
            "<span class='terminal-typing-wrap'>"
            f"<span class='terminal-typing' data-lines='{lines_json}'>{initial_status}</span>"
            "</span>"
            f"<span class='terminal-active' data-states='{states_json}'>"
            f"<span class='active-label'>{html.escape(status_states[0])}</span>"
            "</span>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str, kicker: str = "") -> None:
    st.markdown(
        f"""
        <section class="hero-panel">
            {f'<div class="hero-kicker">{html.escape(kicker)}</div>' if kicker else ''}
            <h1 class="hero-title">{html.escape(title)}</h1>
            <p class="hero-sub2">{html.escape(subtitle)}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def panel(text_html: str) -> None:
    st.markdown(f"<div class='panel'>{text_html}</div>", unsafe_allow_html=True)


def chips(items: Iterable[str], tone: str = "accent") -> str:
    tone_class = {
        "safe": "chip-safe",
        "suspicious": "chip-suspicious",
        "phishing": "chip-phishing",
        "accent": "chip-accent",
    }.get(tone, "chip-accent")
    return "".join([f"<span class='chip {tone_class}'>{html.escape(str(i))}</span>" for i in items])


def risk_meter(score: int, severity: str) -> str:
    s = max(0, min(100, int(score)))
    sev = (severity or "").lower()
    color = "#00d4ff"
    if sev == "low":
        color = "#00ff9f"
    elif sev == "medium":
        color = "#ffdd57"
    elif sev in {"high", "critical"}:
        color = "#ff3860"
    return f"<div class='risk-track'><div class='risk-fill' style='width:{s}%; background:{color}; box-shadow:0 0 10px {color};'></div></div>"
