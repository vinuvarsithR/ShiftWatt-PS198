"""'Ask about this section' assistant.

Answers layman questions about one dashboard section. With a free Gemini key it uses the LLM, grounded ONLY
on that section's computed numbers, and the answer's numbers are fact-checked. Without a key it answers from a
built-in plain-English glossary plus a few data-driven answers, so it always works offline.
"""
from __future__ import annotations

import json
import re

from llm_notes import PROVIDERS, _extract, _fact_numbers, _matches, _parse_json_text, ollama_model, resolve_provider

SYSTEM = """You are ShiftWatt's helper inside an energy dashboard for factory managers and shift supervisors.
Explain things in very simple, friendly language for someone with no technical background (short sentences, everyday
examples, no jargon; if you must use a term, explain it). Keep answers under 100 words. Say "units of electricity" not kWh, "efficiency score" not power factor, "carbon pollution" not CO2 emissions, "very busy/normal/quiet" not load types.
RULES:
- Use ONLY numbers that appear in SECTION DATA. Never invent or calculate new numbers.
- Never state a cause as fact, never predict, never blame workers or specific machines.
- If asked "why" or for a reason: first say the meter data alone cannot prove the cause. Then point out the
  RELATED PATTERNS in SECTION DATA that a manager could look into (for example: one shift spends more time at
  maximum load, has more hours of work, uses more on weekdays than weekends, or has more alerts), quoting those
  numbers. End with 1-2 practical things a manager could check (production schedule, machines running, holidays).
- Shift timings are user-defined assumptions; mention this if relevant.
- Questions about ANYTHING on this page (a chart, a number, a word, a colour, how something is calculated, what to
  look at) or about electricity use in a factory are ALWAYS in scope: answer them using PAGE DESCRIPTION,
  HOW THINGS ARE CALCULATED and SECTION DATA. "Explain the pie chart" means: say what the chart shows and read out
  its key numbers. Only for topics that have nothing to do with this dashboard (e.g. sports, movies) say you can
  only help with this dashboard.
Return JSON: {"answer": "..."}"""

GLOSSARY = {
    ("kwh", "kilowatt hour", "unit", "energy"): "A 'unit' of electricity is the same unit your electricity bill "
        "counts (its technical name is kWh). One unit is what a 1,000-watt heater uses in one hour. More units means "
        "more electricity used and a bigger bill.",
    ("kw ", "kw?", "demand", "what is kw"): "kW is how fast electricity is being used at a moment (like speed), while kWh is "
        "how much was used in total (like distance travelled).",
    ("per hour", "operating hour", "kwh/h", "fair", "'per hour'"): "'Units per hour of work' divides a shift's total "
        "electricity by the hours it ran. It lets you compare shifts fairly even if one shift is longer than another.",
    ("power factor", "pf", "efficiency", "efficient", "score"): "The efficiency score (technical name: power factor) "
        "shows how well the electricity is being used, out of 100. Closer to 100 means less electricity is wasted in "
        "the wires. Electricity companies usually want it to stay high.",
    ("co2", "carbon", "emission", "pollution"): "Carbon pollution (CO2) is the gas released when the electricity you "
        "used was produced at the power plant. More electricity usually means more pollution. It is shown in tonnes.",
    ("load type", "load", "busy", "quiet"): "'How busy' shows, for each shift, how much of its time the factory was "
        "quiet, normal or very busy, based on how much electricity was being drawn at that moment.",
    ("alert", "abnormal", "unusual", "picky", "problem", "spike"): "Unusual usage means electricity use was far "
        "from normal for that same day of the week and time (for example, compared with other Thursday nights at 11 PM). "
        "'Far above normal' can mean waste or extra work; 'far below normal' can mean a stoppage. The 'How picky' choice "
        "decides whether you see only the biggest cases or smaller ones too.",
    ("critical", "warning", "severity", "big", "medium"): "Big problems are the cases furthest from normal; medium ones "
        "are smaller but still unusual. Look at the big ones first.",
    ("typical", "normal", "expected", "baseline"): "'Typical' is the usual reading for that exact weekday and time, "
        "worked out from the whole year of data (the middle value, so one odd day doesn't change it).",
    ("coverage", "missing", "gap", "quality"): "Coverage shows what share of the expected readings are actually in the "
        "file. 100% means no readings are missing, so comparisons are reliable.",
    ("assumption", "shift timing", "shift time"): "Shift timings are typed in by you, not measured by the meter. That is "
        "why every result says they are assumptions: change them in the sidebar and everything recalculates.",
    ("verified", "fact-check", "fact check", "verifier", "tick", "checker"): "After the AI writes the brief, our checker compares every "
        "number it wrote with the numbers we calculated. A tick means every number matched exactly.",
    ("translate", "language", "kannada", "hindi", "voice", "listen"): "Pick a language and press Translate & speak. The "
        "brief is translated, every number is checked again after translation, and then it is read aloud.",
    ("pie", "donut", "circle"): "The round (donut) chart shows each shift's share of all the electricity used. A bigger "
        "slice means that shift used a bigger part of the total. The number in the middle is the total units.",
    ("read this chart", "read the chart", "graph", "chart"): "Each line or bar shows electricity used. Higher means "
        "more electricity. The coloured bands mark your shifts, so you can see which shift is running when usage "
        "goes up or down.",
    ("weekday", "weekend"): "Weekday vs weekend compares the average electricity used per day on Monday–Friday "
        "against Saturday–Sunday, for each shift.",
    ("cost", "rupee", "tariff", "₹", "bill"): "Estimated cost is total kWh multiplied by the tariff you enter in the "
        "sidebar. It is an estimate based on your assumption, not your actual bill.",
}

FRIENDLY = {"total_kwh": ("total electricity use", " units"), "weekday_avg_daily_kwh": ("average use per weekday", " units"),
            "high_alerts": ("number of high-usage alerts", ""), "total_co2": ("carbon pollution", " tonnes"),
            "energy_per_operating_hour": ("use per hour of work", " units")}


def _say(s: dict, key: str) -> str:
    label, unit = FRIENDLY.get(key, (key.replace("_", " "), ""))
    return f"{label}: {s[key]:,}{unit}"


def _text_numbers(obj) -> list[float]:
    """Numbers written inside the page descriptions / explanations we gave the model (e.g. 'score ≥ 8')."""
    out = []
    if isinstance(obj, str):
        out += [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*\.?\d*", obj) if x.replace(",", "").replace(".", "").isdigit()]
    elif isinstance(obj, dict):
        for v in obj.values():
            out += _text_numbers(v)
    elif isinstance(obj, list):
        for v in obj:
            out += _text_numbers(v)
    return out


WHY = re.compile(r"\b(why|reason|cause|because)\b", re.I)


def _offline(question: str, context: dict) -> str:
    q = question.lower()
    if WHY.search(q):
        hints = []
        sh = [x for x in (context.get("shifts") or []) if isinstance(x, dict)]
        mix = [x for x in sh if isinstance(x.get("load_type_mix"), dict)]
        if mix:
            m = max(mix, key=lambda x: x["load_type_mix"].get("Maximum_Load", 0))
            hints.append(f"{m['name']} spends the most time at maximum load "
                         f"({m['load_type_mix'].get('Maximum_Load', 0)}% of its time)")
        wk = [x for x in sh if x.get("weekday_avg_daily_kwh") and x.get("weekend_avg_daily_kwh")]
        if wk:
            w = max(wk, key=lambda x: x["weekday_avg_daily_kwh"] - x["weekend_avg_daily_kwh"])
            hints.append(f"{w['name']} uses much more on weekdays ({w['weekday_avg_daily_kwh']:,} units) than on "
                         f"weekends ({w['weekend_avg_daily_kwh']:,} units), which suggests it follows the work schedule")
        txt = "The meter data alone can't prove the cause. "
        if hints:
            txt += "Related patterns worth looking into: " + "; ".join(hints) + ". "
        return txt + ("A manager could compare this with the production schedule, which machines ran, and holidays "
                      "for those times.")
    shifts = context.get("shifts") or []
    if "summar" in q or "one line" in q:
        if context.get("summary"):
            return context["summary"]
    if "chatgpt" in q or "different" in q or "unlike" in q:
        return ("A normal chatbot reads your file and may guess or miscalculate numbers, and gives a different answer "
                "each time. Here, Python calculates every number exactly, the AI only explains them, and a checker "
                "confirms every number the AI writes. It also works offline and can read the brief in your language.")
    rep = context.get("report")
    if rep and re.search(r"good|reliable|enough|trust", q):
        verdict = "good enough to trust" if rep["coverage_pct"] >= 95 else "usable, but has gaps to keep in mind"
        return (f"Your data covers {rep['coverage_pct']}% of the expected time ({rep['rows']:,} readings, "
                f"{rep['missing_intervals']:,} missing), so it is {verdict}.")
    if rep and "column" in q:
        return "The app uses these columns: " + ", ".join(context.get("columns_used", [])) + "."
    if shifts and "efficien" in q and "avg_power_factor" in shifts[0]:
        low = min(shifts, key=lambda s: s.get("avg_power_factor") or 0)
        return (f"{low['name']} used electricity least efficiently: its efficiency score was "
                f"{low['avg_power_factor']} out of 100, the lowest of all shifts. Closer to 100 is better.")
    if shifts and re.search(r"\b(most|highest|biggest|top|busiest)\b", q):
        key = "total_kwh" if "total_kwh" in shifts[0] else next((k for k in shifts[0] if k != "name"), None)
        if key:
            top = max(shifts, key=lambda s: s.get(key) or 0)
            return f"{top['name']} has the highest {_say(top, key)}."
    if shifts and re.search(r"\b(least|lowest|smallest)\b", q):
        key = "total_kwh" if "total_kwh" in shifts[0] else next((k for k in shifts[0] if k != "name"), None)
        if key:
            low = min(shifts, key=lambda s: s.get(key) or 0)
            return f"{low['name']} has the lowest {_say(low, key)}."
    hits = [a for keys, a in GLOSSARY.items() if any(k in q for k in keys)]
    if hits:
        return " ".join(hits[:2])
    return ("I can explain the terms and charts on this page, for example: kWh, energy per hour, alerts, power factor, "
            "CO2, load type or data coverage. Add a free Gemini key in the sidebar for open-ended questions.")


def ask(question: str, section: str, context: dict, provider: str = "auto") -> dict:
    """Return {"answer", "source", "verified", "unverified"}. Uses Gemini or a local Ollama model when available."""
    prov = resolve_provider(provider)
    if prov in PROVIDERS:
        name = "Gemini" if prov == "gemini" else f"Ollama ({ollama_model()})"
        try:
            user = (f"SECTION: {section}\nPAGE DESCRIPTION, HOW THINGS ARE CALCULATED and SECTION DATA (JSON):\n{json.dumps(context, default=str)[:12000]}\n\n"
                    f"QUESTION: {question}")
            ans = str(_parse_json_text(PROVIDERS[prov](SYSTEM, user))["answer"]).strip()
            nums = _fact_numbers(context) + _text_numbers(context)
            names = [s.get("name", "") for s in context.get("shifts", []) if isinstance(s, dict)]
            bad = [raw for v, raw, _ in _extract(ans, names)
                   if not (0 <= v <= 10 and "." not in raw) and not _matches(v, nums)]
            return {"answer": ans, "source": name, "verified": not bad, "unverified": bad}
        except Exception as e:  # noqa: BLE001
            return {"answer": _offline(question, context), "source": f"built-in ({name} failed: {type(e).__name__})",
                    "verified": True, "unverified": []}
    return {"answer": _offline(question, context), "source": "built-in (no AI model found)", "verified": True,
            "unverified": []}
