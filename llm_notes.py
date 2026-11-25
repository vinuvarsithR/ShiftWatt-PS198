"""Module 5: LLM narrator + number verifier (owner: Kalyan).

Principle: code calculates, the LLM explains, the verifier checks.
Providers (all free): gemini (free tier) | ollama (local) | template (offline, deterministic) | auto.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

CACHE_DIR = Path(__file__).parent / ".llm_cache"

SYSTEM_PROMPT = """You write a short electricity report about factory work shifts for people with NO technical
knowledge (shop-floor supervisors, owners). Use everyday words and short sentences.

WORDING (always use these plain words):
- energy in kWh -> "units of electricity" (1 unit = 1 kWh, like on an electricity bill)
- energy_per_operating_hour -> "units used for every hour of work"
- power factor -> "efficiency score (out of 100, higher is better)"
- CO2 -> "carbon pollution (CO2)" in tonnes
- load types -> Light_Load = "quiet", Medium_Load = "normal", Maximum_Load = "very busy"
- alerts -> "times usage was far above / far below normal"

STRICT RULES
1. Use ONLY numbers that appear in the FACTS JSON. You may round them or add commas. Never calculate new numbers.
2. Never state causes, never predict, never recommend, never blame workers or specific machines.
3. Always say the shift timings were set by the user (they are assumptions).
4. Mention every data-quality warning.

Return ONLY valid JSON:
{"summary": "2 short sentences",
 "observations": [{"title": "2-4 plain words", "text": "1-2 short sentences", "cited_values": [numbers used]}],
 "data_quality_notes": ["..."],
 "assumption_statement": "one sentence saying the shift times were set by the user"}
Write 5-7 observations: who used the most, units per hour of work, weekdays vs weekends, carbon pollution,
efficiency score, how busy each shift was, and (if alerts_summary exists) unusual usage per shift. Skip null values."""

CAUSAL_RE = re.compile(
    r"\b(because|due to|caused|causes|cause|will|likely|should|indicates? a fault|leads? to|"
    r"results? in|resulted in|owing to|attributed to)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------- prompt
def _compact_facts(facts: dict) -> dict:
    f = dict(facts)
    f["anomalies"] = facts.get("anomalies", [])[:5]
    return f


def build_prompt(facts: dict) -> tuple[str, str]:
    user = (
        "FACTS JSON (all numbers are pre-computed; use only these):\n"
        + json.dumps(_compact_facts(facts), indent=1)
        + "\n\nWrite the observation notes as JSON."
    )
    return SYSTEM_PROMPT, user


# ---------------------------------------------------------------- providers
def _call_gemini(system: str, user: str) -> str:
    """Gemini with automatic retries (busy server / rate limit) and backup models.

    Main model: GEMINI_MODEL. Backups: GEMINI_FALLBACKS (comma-separated; filled by 'Find my Gemini models')."""
    import time

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    models = [os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")]
    models += [m for m in os.environ.get("GEMINI_FALLBACKS", "").split(",") if m and m not in models]
    cfg = types.GenerateContentConfig(system_instruction=system, temperature=0.2,
                                      response_mime_type="application/json")
    last = None
    for model in models[:3]:
        for wait in (0, 1, 2, 4):
            if wait:
                time.sleep(wait)
            try:
                resp = client.models.generate_content(model=model, contents=user, config=cfg)
                os.environ["GEMINI_LAST_MODEL"] = model
                return resp.text
            except Exception as e:  # noqa: BLE001
                last = e
                code = getattr(e, "code", None) or getattr(e, "status_code", None)
                busy = type(e).__name__ == "ServerError" or code in (429, 500, 502, 503, 504) or "UNAVAILABLE" in str(e)
                if not busy:
                    break  # wrong key / bad request: retrying will not help, try next model
    raise last


_OLLAMA = {"t": 0.0, "models": []}


def ollama_models(max_age: float = 20.0) -> list[str]:
    """Models installed in a locally running Ollama ([] if Ollama is not running). Cached for a few seconds."""
    import time

    if time.time() - _OLLAMA["t"] < max_age:
        return _OLLAMA["models"]
    models = []
    try:
        import requests

        r = requests.get(os.environ.get("OLLAMA_URL", "http://localhost:11434") + "/api/tags", timeout=1.5)
        models = [m["name"] for m in r.json().get("models", [])]
    except Exception:  # noqa: BLE001
        models = []
    _OLLAMA.update(t=time.time(), models=models)
    return models


def ollama_model() -> str | None:
    """OLLAMA_MODEL if set, else the first installed model (prefers llama models)."""
    if os.environ.get("OLLAMA_MODEL"):
        return os.environ["OLLAMA_MODEL"]
    ms = ollama_models()
    return next((m for m in ms if "llama" in m), ms[0] if ms else None)


def _call_ollama(system: str, user: str) -> str:
    import requests

    model = ollama_model()
    if not model:
        raise RuntimeError("Ollama is not running or has no models (run: ollama pull llama3.2)")
    resp = requests.post(
        os.environ.get("OLLAMA_URL", "http://localhost:11434") + "/api/chat",
        json={
            "model": model,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.2, "num_predict": 900},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        },
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


PROVIDERS = {"gemini": _call_gemini, "ollama": _call_ollama}


def _parse_json_text(text: str) -> dict:
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    return json.loads(t)


def _parse(text: str) -> dict:
    data = _parse_json_text(text)
    if isinstance(data, list):  # some models wrap the object in a list
        data = next((d for d in data if isinstance(d, dict)), {})
    obs = [
        {"title": str(o.get("title", "")), "text": str(o.get("text", "")),
         "cited_values": [v for v in o.get("cited_values", []) if isinstance(v, (int, float))]}
        for o in data.get("observations", []) if isinstance(o, dict)
    ]
    return {
        "summary": str(data.get("summary", "")),
        "observations": obs,
        "data_quality_notes": [str(x) for x in data.get("data_quality_notes", [])],
        "assumption_statement": str(data.get("assumption_statement", "")),
    }


# ---------------------------------------------------------------- offline narrator
def _fmt(v, d=2):
    return f"{v:,.{d}f}"


def template_notes(facts: dict) -> dict:
    """Deterministic plain-language narrator (offline / fallback). Uses only fact values."""
    sh = [s for s in facts["shifts"] if s["readings"]]
    c = facts.get("comparisons", {})
    by = {s["name"]: s for s in sh}
    obs = []
    if c:
        hi, lo = by[c["highest_total"]], by[c["lowest_total"]]
        obs.append({
            "title": "Who used the most",
            "text": f"{hi['name']} used {_fmt(hi['total_kwh'], 0)} units of electricity, which is "
                    f"{_fmt(c['pct_diff_highest_vs_lowest'], 0)}% more than {lo['name']} ({_fmt(lo['total_kwh'], 0)} units).",
            "cited_values": [hi["total_kwh"], c["pct_diff_highest_vs_lowest"], lo["total_kwh"]],
        })
        ph = [by[n] for n in c["ranking_by_per_hour"]]
        obs.append({
            "title": "Units per hour of work",
            "text": "For every hour of work, " + ", ".join(
                f"{s['name']} used {_fmt(s['energy_per_operating_hour'], 0)} units" for s in ph) + ".",
            "cited_values": [s["energy_per_operating_hour"] for s in ph],
        })
    wk = [s for s in sh if s["weekday_avg_daily_kwh"] is not None and s["weekend_avg_daily_kwh"] is not None]
    if wk:
        obs.append({
            "title": "Weekdays vs weekends",
            "text": " ".join(f"{s['name']} used about {_fmt(s['weekday_avg_daily_kwh'], 0)} units on a normal weekday "
                             f"and {_fmt(s['weekend_avg_daily_kwh'], 0)} units on a weekend day." for s in wk),
            "cited_values": [v for s in wk for v in (s["weekday_avg_daily_kwh"], s["weekend_avg_daily_kwh"])],
        })
    if c.get("highest_co2"):
        obs.append({
            "title": "Carbon pollution",
            "text": "Carbon pollution (CO2) from the electricity: " + ", ".join(
                f"{s['name']} {_fmt(s['total_co2'], 1)} tonnes" for s in sh) + ".",
            "cited_values": [s["total_co2"] for s in sh],
        })
    if c.get("lowest_avg_power_factor"):
        lp = by[c["lowest_avg_power_factor"]]
        obs.append({
            "title": "Efficiency score",
            "text": f"{lp['name']} had the lowest efficiency score: {_fmt(lp['avg_power_factor'], 0)} out of 100 "
                    f"(higher is better). " + ", ".join(f"{s['name']}: {_fmt(s['avg_power_factor'], 0)}"
                                                        for s in sh if s is not lp) + ".",
            "cited_values": [s["avg_power_factor"] for s in sh],
        })
    mix = [s for s in sh if s.get("load_type_mix")]
    if mix:
        obs.append({
            "title": "How busy each shift was",
            "text": " ".join(f"{s['name']} was very busy {_fmt(s['load_type_mix'].get('Maximum_Load', 0), 0)}% of the "
                             f"time and quiet {_fmt(s['load_type_mix'].get('Light_Load', 0), 0)}% of the time."
                             for s in mix),
            "cited_values": [v for s in mix for v in s["load_type_mix"].values()],
        })
    al = facts.get("alerts_summary")
    if al:
        obs.append({
            "title": "Unusual usage",
            "text": " ".join(f"{n}: {v['high_alerts']} times far above normal and {v['low_alerts']} times far below "
                             f"normal." for n, v in al.items()),
            "cited_values": [x for v in al.values() for x in v.values()],
        })
    summary = (
        f"Over {_fmt(facts['data_period']['days'], 0)} days, {c.get('highest_total', '-')} used the most electricity "
        f"and {c.get('lowest_total', '-')} used the least. This shows how the shifts differ; it does not show why."
    )
    return {
        "summary": summary,
        "observations": obs,
        "data_quality_notes": list(facts.get("data_quality_warnings", [])) or ["No problems found in the data."],
        "assumption_statement": "The shift times were set by you, so they are assumptions, not measured.",
    }


# ---------------------------------------------------------------- verifier
_TIME = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b")
_NUM = re.compile(r"(?<![\w.])-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.])-?\d+(?:\.\d+)?")


def _fact_numbers(obj) -> list[float]:
    out = []
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            out += _fact_numbers(v)
    elif isinstance(obj, list):
        for v in obj:
            out += _fact_numbers(v)
    return out


def _extract(text: str, shift_names: list[str]):
    clean = text
    for n in sorted(shift_names, key=len, reverse=True):
        clean = clean.replace(n, " ")
    clean = _DATE.sub(" ", _TIME.sub(" ", clean))
    for m in _NUM.finditer(clean):
        raw = m.group()
        yield float(raw.replace(",", "")), raw, clean[max(0, m.start() - 40): m.end() + 20].strip()


def _matches(v: float, facts_nums: list[float]) -> bool:
    return any(abs(v - f) <= max(0.005 * abs(f), 0.051 if v != int(v) else 0.5) for f in facts_nums)


def _pools(facts: dict):
    """Numbers that belong to each shift, and numbers that are not tied to one shift."""
    per = {}
    for sh in facts.get("shifts", []):
        per.setdefault(sh.get("name"), []).extend(_fact_numbers({k: v for k, v in sh.items() if k != "name"}))
    for name, v in (facts.get("alerts_summary") or {}).items():
        per.setdefault(name, []).extend(_fact_numbers(v))
    for an in facts.get("anomalies", []):
        per.setdefault(an.get("shift"), []).extend(_fact_numbers({k: v for k, v in an.items() if k != "shift"}))
    glob = _fact_numbers({k: v for k, v in facts.items() if k not in ("shifts", "alerts_summary", "anomalies")}) + [100.0]
    glob += [float(y) for y in re.findall(r"\b(19\d\d|20\d\d)\b", json.dumps(facts.get("data_period", {})))]  # years
    return per, glob


def _mask(text: str) -> str:
    """Blank out times and dates (same length, so positions stay valid)."""
    for rx in (_DATE, _TIME):
        text = rx.sub(lambda m: " " * len(m.group()), text)
    return text


def verify_notes(notes: dict, facts: dict, scoped: bool = True) -> dict:
    """Check every number in the AI text against the computed facts and flag causal/predictive words.

    scoped=True (default): a number written next to a shift name must match THAT shift's values (or a value not tied
    to any shift, e.g. a comparison). This also catches real numbers attached to the wrong shift.
    scoped=False: the older check against all computed values (kept for benchmarking)."""
    per, glob = _pools(facts)
    nums = _fact_numbers(facts) + [g for g in glob if g >= 1900]
    names = sorted([s["name"] for s in facts.get("shift_definitions", [])], key=len, reverse=True)
    checked, mismatches, flagged = 0, [], []

    def allowed(v: float, sentence: str, pos: int) -> bool:
        if not scoped:
            return _matches(v, nums)
        hits = sorted((m.start(), n) for n in names for m in re.finditer(re.escape(n), sentence))
        if not hits:
            return _matches(v, nums)
        # the clause around the number (split on ", " / "; " – not inside "470,188")
        c0 = max(sentence.rfind(", ", 0, pos), sentence.rfind("; ", 0, pos)) + 1
        nxt = [i for i in (sentence.find(", ", pos), sentence.find("; ", pos)) if i != -1]
        c1 = min(nxt) if nxt else len(sentence)
        before = [n for st, n in hits if c0 <= st < pos]
        after = [n for st, n in hits if pos <= st < c1]
        if not before:  # owner unclear, e.g. a list "89, 124 and 55 times for Shift A, B and C"
            return _matches(v, nums)
        return _matches(v, glob) or _matches(v, per.get(before[-1], []))

    def check(text: str) -> bool:
        nonlocal checked
        ok = True
        for sent in re.split(r"(?<=[.!?])\s+", text):
            masked = _mask(sent)
            for n in names:  # hide digits inside shift names ("Shift 1")
                masked = masked.replace(n, "#" * len(n))
            for m in _NUM.finditer(masked):
                raw = m.group()
                v = float(raw.replace(",", ""))
                if 0 <= v <= 10 and "." not in raw:  # small counts / ordinals ("3 shifts")
                    continue
                checked += 1
                if not allowed(v, sent, m.start()):
                    ok = False
                    mismatches.append({"value": raw, "context": sent[max(0, m.start() - 40): m.end() + 20].strip()})
        for m in CAUSAL_RE.finditer(text):
            ok = False
            flagged.append(text[max(0, m.start() - 30): m.end() + 30].strip())
        return ok

    check(notes.get("summary", ""))
    for o in notes.get("observations", []):
        o["verified"] = check(o.get("title", "") + ". " + o.get("text", ""))
    result = {
        "all_verified": not mismatches and not flagged,
        "checked": checked,
        "mismatches": mismatches,
        "flagged_phrases": flagged,
    }
    notes["verification"] = result
    return result


# ---------------------------------------------------------------- main entry
def _cache_key(facts, provider):
    return hashlib.sha256((json.dumps(facts, sort_keys=True) + provider).encode()).hexdigest()[:16]


def resolve_provider(provider: str = "auto") -> str:
    if provider != "auto":
        return provider
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini"
    if ollama_models():
        return "ollama"
    return "template"


def generate_notes(facts: dict, provider: str = "auto", use_cache: bool = True) -> dict:
    provider = resolve_provider(provider)
    meta = {"provider": provider, "cached": False, "error": None}
    key = _cache_key(facts, provider)
    path = CACHE_DIR / f"{key}.json"

    notes = None
    if provider != "template" and use_cache and path.exists():
        notes = json.loads(path.read_text())
        meta["cached"] = True
    elif provider in PROVIDERS:
        system, user = build_prompt(facts)
        for attempt in range(2):  # one retry on invalid JSON / API error
            try:
                notes = _parse(PROVIDERS[provider](system, user))
                if notes["observations"]:
                    break
            except Exception as e:  # noqa: BLE001
                meta["error"] = f"{type(e).__name__}: {e}"[:300]
                notes = None
        if notes:
            meta["error"] = None
            CACHE_DIR.mkdir(exist_ok=True)
            path.write_text(json.dumps(notes))

    if notes is None:
        if provider != "template":
            meta["provider"] = f"template (fallback from {provider})"
        notes = template_notes(facts)

    if not notes.get("assumption_statement"):
        notes["assumption_statement"] = facts["assumption_note"]
    verify_notes(notes, facts)
    notes["meta"] = meta
    return notes


def notes_to_markdown(notes: dict, facts: dict) -> str:
    L = ["# Manufacturing Shift Trend Brief", "",
         f"> **Assumption:** {notes['assumption_statement']}", "",
         f"Data period: {facts['data_period']['start']} to {facts['data_period']['end']} "
         f"({facts['data_period']['days']} days)", "",
         "## Shift definitions (user assumptions)", "", "| Shift | Start | End | Hours |", "|---|---|---|---|"]
    L += [f"| {s['name']} | {s['start']} | {s['end']} | {s['duration_hours']:g} |" for s in facts["shift_definitions"]]
    L += ["", "## Shift comparison", "",
          "| Shift | Total kWh | kWh per operating hour | Peak kWh | Coverage % | CO2 (t) | Avg PF % |",
          "|---|---|---|---|---|---|---|"]
    for s in facts["shifts"]:
        L.append(f"| {s['name']} | {s['total_kwh']} | {s['energy_per_operating_hour']} | {s['peak_kwh']} | "
                 f"{s['coverage_pct']} | {s['total_co2']} | {s['avg_power_factor']} |")
    L += ["", "## Summary", "", notes["summary"], "", "## Observations", ""]
    for o in notes["observations"]:
        mark = "verified" if o.get("verified") else "CHECK"
        L.append(f"- **{o['title']}** ({mark}): {o['text']}")
    L += ["", "## Data quality", ""] + [f"- {n}" for n in notes["data_quality_notes"]]
    v = notes.get("verification", {})
    L += ["", "## Verification", "",
          f"{v.get('checked', 0)} numbers checked against computed values; "
          f"{len(v.get('mismatches', []))} mismatches; {len(v.get('flagged_phrases', []))} flagged phrases.", "",
          "_Observations describe recorded data only. They do not establish causes or predict future use._"]
    return "\n".join(L)
