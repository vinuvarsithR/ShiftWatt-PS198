"""Regional-language briefs: translate (Gemini free tier, or bundled offline translations),
re-verify every number after translation, and speak it (gTTS free MP3, or browser voices).
"""
from __future__ import annotations

import hashlib
import io
import json
import re
from pathlib import Path

from llm_notes import _call_gemini, _extract, _matches, _parse_json_text

LANGS = {
    "English": ("en", "English"),
    "हिन्दी · Hindi": ("hi", "Hindi"),
    "ಕನ್ನಡ · Kannada": ("kn", "Kannada"),
    "தமிழ் · Tamil": ("ta", "Tamil"),
    "తెలుగు · Telugu": ("te", "Telugu"),
    "മലയാളം · Malayalam": ("ml", "Malayalam"),
    "मराठी · Marathi": ("mr", "Marathi"),
    "বাংলা · Bengali": ("bn", "Bengali"),
    "ગુજરાતી · Gujarati": ("gu", "Gujarati"),
}
PERCENT = {"en": " percent", "hi": " प्रतिशत", "kn": " ಶೇಕಡಾ", "ta": " சதவீதம்", "te": " శాతం", "ml": " ശതമാനം",
           "mr": " टक्के", "bn": " শতাংশ", "gu": " ટકા"}
OFFLINE_DIR = Path(__file__).parent / "translations"
CACHE_DIR = Path(__file__).parent / ".llm_cache"

SYSTEM = """You translate factory energy reports for shift supervisors in India.
Translate the text into {lang}, in simple, natural spoken language that a shop-floor supervisor understands.
STRICT RULES:
- Keep EVERY number exactly as written, using Western digits (0-9), including decimals and commas. Never convert, round or drop a number.
- Keep shift names (e.g. "Shift A") recognisable; you may transliterate the word "Shift".
- Translate units into words in {lang} (e.g. kWh -> kilowatt-hour). Keep "CO2" as is.
- Do not add, remove or reinterpret any information. Do not add causes or advice.
- Keep one output line per input line.
Return JSON: {{"translation": "<translated text, lines separated by \\n>"}}"""


def brief_text(notes: dict) -> str:
    """The spoken brief: assumption, summary, then one line per observation."""
    lines = [notes["assumption_statement"], notes["summary"]]
    lines += [f"{o['title']}: {o['text']}" for o in notes["observations"]]
    return "\n".join(lines)


def _key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def translate(text: str, code: str, name: str) -> tuple[str | None, str]:
    """Return (translated_text, source). Source: original | offline | cache | gemini | unavailable: <reason>."""
    if code == "en":
        return text, "original"
    k = _key(text)
    for d, src in ((OFFLINE_DIR, "offline"), (CACHE_DIR, "cache")):
        f = d / f"{k}_{code}.txt"
        if f.exists():
            return f.read_text(encoding="utf-8"), src
    try:
        raw = _call_gemini(SYSTEM.format(lang=name), text)
        out = _parse_json_text(raw)["translation"].strip()
        CACHE_DIR.mkdir(exist_ok=True)
        (CACHE_DIR / f"{k}_{code}.txt").write_text(out, encoding="utf-8")
        return out, "gemini"
    except Exception as e:  # noqa: BLE001
        return None, f"unavailable: {type(e).__name__}"


def verify_translation(source: str, translated: str, shift_names: list[str]) -> dict:
    """Every number in the translation must exist in the source, and no source number may be dropped."""
    src = [v for v, raw, _ in _extract(source, shift_names) if not (0 <= v <= 10 and "." not in raw)]
    dst = [(v, raw, ctx) for v, raw, ctx in _extract(translated, shift_names) if not (0 <= v <= 10 and "." not in raw)]
    dst_vals = [v for v, _, _ in dst]
    changed = [{"value": raw, "context": ctx} for v, raw, ctx in dst if not _matches(v, src)]
    dropped = [v for v in src if not _matches(v, dst_vals)]
    return {"ok": not changed and not dropped, "checked": len(dst), "changed": changed, "dropped": dropped}


def speech_ready(text: str, code: str) -> str:
    """Round long numbers and spell units/percent so the voice sounds natural."""
    def num(m):
        raw = m.group()
        v = float(raw.replace(",", ""))
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        return f"{v:.1f}".rstrip("0").rstrip(".") if "." in raw else raw
    t = re.sub(r"(?<![A-Za-z])\d[\d,]*\.?\d*", num, text)
    t = t.replace("%", PERCENT.get(code, " percent"))
    if code == "en":
        for a, b in [("kWh/h", "kilowatt hours per hour"), ("kWh", "kilowatt hours"), ("CO2", "C O 2")]:
            t = t.replace(a, b)
    return t


def tts_mp3(text: str, code: str) -> bytes | None:
    """Free Google text-to-speech MP3 (needs internet, no key). None if unavailable."""
    try:
        from gtts import gTTS

        buf = io.BytesIO()
        gTTS(text=text, lang=code, slow=False).write_to_fp(buf)
        return buf.getvalue()
    except Exception:  # noqa: BLE001
        return None
