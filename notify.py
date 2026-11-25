"""Alert notifications: on-screen pop-up (st.toast, in app.py), desktop notification + beep (browser), and an
optional free phone push via a Telegram bot."""
from __future__ import annotations

import json
import threading

import streamlit as st
import streamlit.components.v1 as components


def _frame(html: str, height: int):
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:
        components.html(html, height=height)


def enable_button(dark: bool):
    """Button that asks the browser for permission to show desktop notifications."""
    fg, bg, line = ("#ECEDEE", "#2E3237", "#3A3F45") if dark else ("#1F2329", "#FFFFFF", "#D1D5DB")
    _frame(f"""<div style="font-family:Inter,Segoe UI,Arial;color:{fg};font-size:14px;display:flex;gap:10px;align-items:center">
<button id="b" style="background:{bg};color:{fg};border:1px solid {line};border-radius:10px;padding:8px 14px;cursor:pointer">
🔔 Turn on desktop notifications</button><span id="s" style="opacity:.75"></span></div>
<script>
const W = window.parent || window, s = document.getElementById("s");
function show() {{ s.textContent = !("Notification" in W) ? "Not supported in this browser" :
  W.Notification.permission === "granted" ? "✓ Desktop notifications are on" :
  W.Notification.permission === "denied" ? "Blocked – allow notifications for this site in the browser" : ""; }}
show();
document.getElementById("b").onclick = () => {{ if ("Notification" in W) W.Notification.requestPermission().then(show); }};
</script>""", 48)


def desktop_alert(title: str, body: str, uid: str):
    """Fire a desktop notification + short beep (runs once per unique uid)."""
    _frame(f"""<script>
const W = window.parent || window;
try {{
  const AC = W.AudioContext || W.webkitAudioContext;
  if (AC) {{ const c = new AC(); [0, 0.25].forEach(t => {{ const o = c.createOscillator(), g = c.createGain();
    o.frequency.value = 880; o.connect(g); g.connect(c.destination); g.gain.value = 0.15;
    o.start(c.currentTime + t); o.stop(c.currentTime + t + 0.15); }}); }}
}} catch (e) {{}}
try {{ if ("Notification" in W && W.Notification.permission === "granted")
  new W.Notification({json.dumps(title)}, {{ body: {json.dumps(body)}, tag: {json.dumps(uid)} }}); }} catch (e) {{}}
</script>""", 1)


def send_telegram(token: str, chat_id: str, text: str, wait: bool = False) -> str | None:
    """Free phone push via a Telegram bot. Runs in the background unless wait=True (returns an error or None)."""
    def go():
        import requests
        try:
            r = requests.post(f"https://api.telegram.org/bot{token.strip()}/sendMessage",
                              json={"chat_id": chat_id.strip(), "text": text}, timeout=8)
            return None if r.ok else r.json().get("description", f"HTTP {r.status_code}")
        except Exception as e:  # noqa: BLE001
            return f"{type(e).__name__}"
    if not (token and chat_id):
        return "Telegram not set up"
    if wait:
        return go()
    threading.Thread(target=go, daemon=True).start()
    return None
