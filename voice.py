"""Voice read-out using the browser's built-in Web Speech API (free, offline, no API key, no install)."""
from __future__ import annotations

import json
import re

import streamlit as st
import streamlit.components.v1 as components


def speech_text(text: str) -> str:
    """Make report text sound natural: round long numbers, spell out units."""
    def num(m):
        v = float(m.group().replace(",", ""))
        return f"{v:,.0f}" if abs(v) >= 1000 else (f"{v:.1f}".rstrip("0").rstrip(".") if "." in m.group() else m.group())
    t = re.sub(r"\d[\d,]*\.?\d*", num, text)
    for a, b in [("kWh/h", "kilowatt hours per hour"), ("kWh", "kilowatt hours"), ("tCO₂", "tonnes of C O 2"),
                 ("CO₂", "C O 2"), ("CO2", "C O 2"), ("%", " percent"), ("–", " to "), ("₹", " rupees "),
                 ("_", " "), ("**", ""), ("`", "")]:
        t = t.replace(a, b)
    return t


def voice_player(text: str, key: str, label: str = "🔊 Read aloud", lang: str = "en", dark: bool = False) -> None:
    """Browser voices. For non-English, pass already speech-ready text and the language code (hi, kn, ...)."""
    spoken = speech_text(text) if lang == "en" else text
    sentences = [s.strip() for s in re.split(r"(?<=[.!?।])\s+|\n+", spoken) if s.strip()]
    payload = json.dumps(sentences).replace("</", "<\\/")  # safe inside <script>
    fg, bg2, line = ("#ECEDEE", "#2E3237", "#3A3F45") if dark else ("#1F2329", "#FFFFFF", "#D1D5DB")
    html = f"""
<div style="color:{fg};font-family:Inter,Segoe UI,Arial,sans-serif;display:flex;gap:8px;align-items:center;flex-wrap:wrap;font-size:14px">
  <button id="play-{key}" style="background:#22C55E;color:#fff;border:0;border-radius:8px;padding:8px 14px;cursor:pointer;font-size:14px">{label}</button>
  <button id="pause-{key}" style="border:1px solid {line};background:{bg2};color:{fg};border-radius:8px;padding:8px 12px;cursor:pointer">⏯ Pause / resume</button>
  <button id="stop-{key}" style="border:1px solid {line};background:{bg2};color:{fg};border-radius:8px;padding:8px 12px;cursor:pointer">⏹ Stop</button>
  <select id="voice-{key}" style="padding:6px;border-radius:6px;max-width:240px;background:{bg2};color:{fg};border:1px solid {line}"></select>
  <label>Speed <input id="rate-{key}" type="range" min="0.6" max="1.6" step="0.1" value="1"></label>
  <span id="status-{key}" style="opacity:.7"></span>
</div>
<script>
const parts = {payload};
const synth = window.speechSynthesis;
const sel = document.getElementById("voice-{key}");
const status = document.getElementById("status-{key}");
function loadVoices() {{
  const vs = synth.getVoices().filter(v => v.lang.startsWith("{lang}"));
  if (!vs.length) return;
  sel.innerHTML = vs.map((v, i) => `<option value="${{v.name}}">${{v.name}} (${{v.lang}})</option>`).join("");
  const pref = vs.find(v => v.lang === "{lang}-IN") || vs.find(v => v.lang === "en-GB") || vs[0];
  sel.value = pref.name;
}}
if (!synth) {{ status.textContent = "Voice not supported in this browser"; }}
else {{
  loadVoices(); synth.onvoiceschanged = loadVoices;
  document.getElementById("play-{key}").onclick = () => {{
    synth.cancel();
    if (!synth.getVoices().some(v => v.lang.startsWith("{lang}"))) {{
      status.textContent = "No voice for this language on this computer – try Chrome or Edge"; }}
    const voice = synth.getVoices().find(v => v.name === sel.value);
    const rate = parseFloat(document.getElementById("rate-{key}").value);
    parts.forEach((p, i) => {{
      const u = new SpeechSynthesisUtterance(p);
      if (voice) u.voice = voice; u.rate = rate;
      u.onstart = () => status.textContent = `Speaking ${{i + 1}} / ${{parts.length}}`;
      if (i === parts.length - 1) u.onend = () => status.textContent = "Done";
      synth.speak(u);
    }});
  }};
  document.getElementById("pause-{key}").onclick = () => synth.paused ? synth.resume() : synth.pause();
  document.getElementById("stop-{key}").onclick = () => {{ synth.cancel(); status.textContent = "Stopped"; }};
}}
</script>"""
    if hasattr(st, "iframe"):  # Streamlit >= 1.5x
        st.iframe(html, height=60)
    else:
        components.html(html, height=60)
