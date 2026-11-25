"""ShiftWatt – a simple electricity dashboard for factory shifts (PS 198).

Run:  streamlit run app.py
"""
from __future__ import annotations

import datetime as dt
import html
import io
import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from alerts import alert_summary, day_view, detect_alerts, score_readings
from assistant import ask
from data_pipeline import load_and_clean
from llm_notes import generate_notes, notes_to_markdown, ollama_model, ollama_models, resolve_provider
from shift_engine import (DEFAULT_SHIFTS, _minutes, assign_shifts, compute_shift_stats, daily_shift_totals,
                          validate_shifts)
from translate import LANGS, brief_text, speech_ready, translate, tts_mp3, verify_translation
from notify import desktop_alert, enable_button, send_telegram
from voice import voice_player

SAMPLE = Path(__file__).parent / "data" / "Steel_industry_data.csv"
SHIFT_COLORS = ["#1BA3B8", "#F5C242", "#E0245E", "#22C55E"]  # teal, yellow, pink, green
PICKY = {"Only big problems": 8.0, "Big and medium problems": 5.0, "Everything unusual": 3.5}
INFO = {"evalinfo": "**How these tests work**\n\n- **Accuracy**: shift totals are recounted row by row with a separate simple method and compared.\n- **AI mistakes**: we take a correct summary and deliberately damage it 300 times per type (change a number by 2–50%, swap two shifts' numbers, add an invented cause) and count how often the checker catches it. We also check 300 correct summaries to make sure it does not raise false alarms.\n- **Unusual usage**: we add 204 fake problems (sudden jumps of +25 to +60 units, or drops to near zero during normally busy times) at random times in the real data and count how many each method finds.\n- **Speed**: time to process all 35,040 readings on this computer.", "live": "**How it works**\n\n- We replay the recorded 15-minute readings of the chosen day, a few per second, as if they were arriving from the meter right now.\n- **Your limit**: an alert fires the moment a reading goes **above** the number you set (units per 15 minutes).\n- **Far from normal** (optional): an alert also fires when a reading is far from the normal value for that weekday and time, using the 'How picky' setting.\n- An alert fires once when the line crosses; it fires again only after usage comes back down and crosses again.\n- With a real meter feed, the same checks would run on each new reading.", **{'summary': "**How it's calculated**\n\n- **Units** = all 15-minute meter readings (kWh, the same 'unit' as on your bill) that fall inside the shift's times, added up over the whole period.\n- **Trend line** = the shift's total for each month.\n- **Busiest month** = the month with the highest total for that shift.\n- *Uses the most / least* compares the shift totals.", 'table': "**How it's calculated**\n\n- **Share** = shift's units ÷ units of all shifts × 100.\n- **Units per hour of work** = shift's units ÷ hours the shift ran in the period (number of 15-minute slots × ¼ hour). This is fair even if shifts have different lengths.\n- **Weekends use X% less** = 1 − (weekend daily average ÷ weekday daily average), all shifts together.", 'donut': "**How it's calculated**\n\nEach slice = that shift's units ÷ total units of all shifts × 100. The centre shows the total units.", 'picky': "**How it's calculated**\n\n1. **Normal** = the middle value (median) of all readings at the same weekday and 15-minute time (e.g. every Thursday 11:00–11:15 PM) over the whole period.\n2. **Usual spread** = how much those readings normally vary (1.4826 × median absolute difference; at least 5% of normal or 0.5 units).\n3. **Score** = (reading − normal) ÷ usual spread.\n4. A reading is unusual when the score is beyond the setting below **and** it differs from normal by at least 5 units:\n   - Only big problems: score ≥ 8\n   - Big and medium: score ≥ 5\n   - Everything unusual: score ≥ 3.5\n5. Unusual readings back-to-back in the same shift and direction are grouped into one case.", 'biggest': "**How it's calculated**\n\nCases are sorted by how far their most extreme reading was from normal (highest score first).\n\n- **Times normal** = that reading ÷ the normal value for that weekday and time.\n- **% less** = 1 − (reading ÷ normal).", 'day': "**How it's calculated**\n\n- **Blue line**: actual units in each 15 minutes of the chosen day.\n- **Grey dashed line**: the normal (median) value for the same weekday and time, from the whole period.", 'normalday': "**How it's calculated**\n\nFor each 15-minute time of day, the **average** of all readings at that time, separately for weekdays (Mon–Fri) and weekends (Sat–Sun). Coloured areas show your shift times.", 'wkwe': "**How it's calculated**\n\nFor each shift: add up its units for every calendar day, then take the **average across weekdays** and **across weekend days**. A night shift's units are counted on the date each 15 minutes starts.", 'carbon': "**How it's calculated**\n\nThe sample file includes a **CO2(tCO2)** column: tonnes of CO₂ for the electricity used in each 15 minutes. We add these up for each shift. (We do not estimate it ourselves.)", 'eff': "**How it's calculated**\n\nThe file includes a **power factor** reading (0–100) for every 15 minutes, recorded by the meter. The efficiency score is the **average** of these readings during each shift. Higher means less electricity wasted.", 'busy': "**How it's calculated**\n\nWe don't guess this. The sample file has a **Load_Type** column where every 15-minute reading is already labelled **Light**, **Medium** or **Maximum** load by the data source. We show these as Quiet, Normal and Very busy.\n\nFor each shift: number of its readings with each label ÷ all its readings × 100.", 'bill': "**How it's calculated**\n\nTotal units (all shifts) × your price per unit (set in Settings). It is an estimate: it leaves out fixed charges, demand charges and taxes on a real bill.", 'unusual': "**How it's calculated**\n\nNumber of unusual cases found at your current 'How picky' setting (see the 🔔 Unusual usage page).", 'co2tile': "**How it's calculated**\n\nSum of the CO₂ column in the file for all readings inside your shifts, in tonnes.", 'readings': "**How it's calculated**\n\nNumber of rows in the file after removing exact duplicates and rows without a valid time.", 'every': "**How it's calculated**\n\nThe most common gap between two readings in the file.", 'complete': "**How it's calculated**\n\nReadings found ÷ readings expected between the first and last time in the file × 100. Missing readings are never filled in.", 'summarytext': "**How it's checked**\n\nThe AI only receives numbers we already calculated. After it writes, every number in its text is compared with our numbers (allowed difference: 0.5% for rounding). Words that guess causes or predict (like 'because', 'will') are also flagged."}}
PAGE_DESC = {
    "home": ("HOME page shows: (1) Summary cards, one per shift, with total units and a small monthly trend line and "
             "busiest month; (2) a Shift summary table with units used, share %, units per hour of work, and 3 takeaways; "
             "(3) a donut (pie) chart 'Who used the electricity?' where each slice is a shift's share of all units and the "
             "centre shows the total; (4) tiles: estimated bill, number of unusual-usage cases, carbon pollution.",
             ["summary", "table", "donut", "bill", "unusual", "co2tile"]),
    "alerts": ("UNUSUAL USAGE page shows: a 'How picky should we be?' choice, counts of times usage was far MORE and far "
               "LESS than normal, a list of the biggest cases, and a 'Look at one day' line chart (blue = that day, "
               "grey dashed = a normal day like it).", ["picky", "biggest", "day"]),
    "live": ("LIVE MONITOR page replays recorded meter readings one by one as if they were arriving live. When a reading "
             "goes above the user's limit, or is far from normal, an alert pops up, a desktop notification and beep play, "
             "and (if set up) a phone message is sent.", ["live"]),
    "pattern": ("DAILY PATTERN page shows: a line chart of a normal day (average units every 15 minutes through the day, "
                "weekday vs weekend, with coloured bands for each shift) and a bar chart of average units per day on "
                "weekdays vs weekends for each shift.", ["normalday", "wkwe"]),
    "co2": ("POLLUTION & EFFICIENCY page shows bar charts of carbon pollution (tonnes CO2) and efficiency score (out of "
            "100) per shift, and a bar showing how much of each shift's time was quiet, normal or very busy.",
            ["carbon", "eff", "busy"]),
    "brief": ("WRITTEN SUMMARY page: a button writes a plain-language summary with an AI; every number is double-checked; "
              "it can be played aloud in English or 8 Indian languages.", ["summarytext"]),
    "eval": ("EVALUATION page shows test results: (1) calculation accuracy vs an independent recount, (2) how often "
             "the checker catches AI mistakes, before (v1) and after (v2) our improvement, (3) how well unusual usage "
             "is detected compared with two simpler methods, in quiet and busy hours, (4) translation safety, (5) speed, "
             "(6) a live comparison of AI models and prompt versions, and (7) the responsible-AI safeguards.",
             ["evalinfo"]),
    "data": ("DATA CHECK page shows whether the meter data is complete: number of readings, how often a reading is "
             "taken, and the % of expected readings present.", ["readings", "every", "complete"]),
}
PAGES = ["🏠  Home", "🔔  Unusual usage", "📡  Live monitor", "🕒  Daily pattern", "🌱  Pollution & efficiency",
         "📝  Written summary", "✅  Data check", "📊  Evaluation", "⚙️  Settings"]

st.set_page_config(page_title="ShiftWatt", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

# Deployment: read keys from Streamlit secrets (cloud) or environment variables (Docker), never from code.
try:
    for _k in ("GEMINI_API_KEY", "GEMINI_MODEL", "OLLAMA_URL", "OLLAMA_MODEL"):
        if _k in st.secrets and not os.environ.get(_k):
            os.environ[_k] = str(st.secrets[_k])
except Exception:  # noqa: BLE001  (no secrets file is fine)
    pass

# ------------------------------------------------------------------ persistent settings
if "cfg" not in st.session_state:
    st.session_state.cfg = {
        "shifts": [{"name": s["name"], "start": dt.time(*map(int, s["start"].split(":"))),
                    "end": dt.time(*map(int, s["end"].split(":")))} for s in DEFAULT_SHIFTS],
        "price": 8.0, "picky": "Big and medium problems", "upload": None, "upload_name": None,
        "provider": "auto", "label_end": True,
    }
cfg = st.session_state.cfg
dark = st.session_state.get("dark", True)

T = (dict(bg="#1B1D21", panel="#25282D", panel2="#2E3237", side="#141518", text="#ECEDEE", muted="#9BA1A6",
          line="#373B41", grid="#33373D") if dark else
     dict(bg="#F3F4F6", panel="#FFFFFF", panel2="#F5F6F8", side="#FFFFFF", text="#1F2329", muted="#6B7280",
          line="#E5E7EB", grid="#EEF0F3"))

st.markdown(f"""<style>
:root {{ --bg:{T['bg']}; --panel:{T['panel']}; --panel2:{T['panel2']}; --side:{T['side']}; --text:{T['text']};
        --muted:{T['muted']}; --line:{T['line']}; --accent:#22C55E; }}
html, body, .stApp, [data-testid="stAppViewContainer"] {{ background: var(--bg) !important; }}
.stApp, .stApp p, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp td, .stApp th,
[data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"] p {{ color: var(--text) !important;
  font-family: Inter, "Segoe UI", Roboto, Arial, sans-serif; }}
[data-testid="stSidebarCollapseButton"], [data-testid="stSidebarCollapsedControl"] {{ display: none !important; }}
[data-testid="stHeader"] {{ background: transparent !important; }}
[data-testid="stToolbar"], [data-testid="stDecoration"] {{ display: none !important; }}
.block-container {{ padding-top: 1.4rem; max-width: 1250px; }}
[data-testid="stSidebar"] {{ background: var(--side) !important; border-right: 1px solid var(--line); }}
[data-testid="stSidebar"] [role="radiogroup"] {{ gap: 4px; }}
[data-testid="stSidebar"] [role="radiogroup"] label {{ padding: 11px 14px; border-radius: 10px; width: 100%;
  margin: 0; cursor: pointer; display: flex; }}
[data-testid="stSidebar"] [role="radiogroup"] label > div > div:first-child {{ display: none !important; }}
[data-testid="stSidebar"] [role="radiogroup"] label p {{ font-size: 15.5px; }}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {{ background: var(--panel2); }}
[data-testid="stSidebar"] [role="radiogroup"] label[data-selected="true"] {{ background: var(--accent); }}
[data-testid="stSidebar"] [role="radiogroup"] label[data-selected="true"] p {{ color: #fff !important; font-weight: 600; }}
[data-testid="stVerticalBlock"][data-test-scroll-behavior], [data-testid="stVerticalBlockBorderWrapper"] {{
  background: var(--panel); border: 1px solid var(--line) !important; border-radius: 16px; padding: 18px 20px; }}
div[data-baseweb="input"], div[data-baseweb="base-input"], div[data-baseweb="select"] > div, input, textarea {{
  background: var(--panel2) !important; color: var(--text) !important; border-color: var(--line) !important; }}
[data-baseweb="popover"] ul, [data-baseweb="popover"] li, [data-baseweb="menu"] {{ background: var(--panel2) !important;
  color: var(--text) !important; }}
[data-testid="stChatInput"] > div {{ background: var(--panel2) !important; border: 1px solid var(--line) !important; }}
[data-testid="stChatInput"] textarea::placeholder {{ color: var(--muted) !important; }}
button[kind="secondary"], [data-testid^="stBaseButton-secondary"], [data-testid="stDownloadButton"] button {{
  background: var(--panel2) !important; color: var(--text) !important; border: 1px solid var(--line) !important;
  border-radius: 10px !important; }}
button[kind="primary"], [data-testid^="stBaseButton-primary"] {{ background: var(--accent) !important;
  border: none !important; border-radius: 10px !important; color: #fff !important; }}
[data-testid="stPopoverBody"] {{ background: var(--panel) !important; border: 1px solid var(--line) !important;
  max-width: 420px; }}
[data-testid="stPopover"] button {{ border-radius: 999px !important; }}
[data-testid="stExpander"] details {{ background: var(--panel); border: 1px solid var(--line); border-radius: 12px; }}
[data-testid="stChatMessage"] {{ background: var(--panel2); border-radius: 12px; }}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{ color: var(--muted) !important; }}
[data-testid="stFileUploaderDropzone"] {{ background: var(--panel2) !important; }}
.stApp .page-title {{ font-size: 30px !important; font-weight: 800 !important; margin: 0; line-height: 1.2; }}
.stApp .page-sub {{ color: var(--muted) !important; font-size: 15px !important; margin: 4px 0 0; }}
.stApp .card-title {{ font-size: 21px !important; font-weight: 700 !important; margin: 0 0 2px; }}
.stApp .card-sub {{ color: var(--muted) !important; font-size: 14.5px !important; margin-bottom: 10px; }}
.stApp .mini-label {{ color: var(--muted) !important; font-size: 14px; }}
.mini-head {{ font-size: 17px; font-weight: 600; }}
.big {{ font-size: 30px; font-weight: 700; line-height: 1.15; margin: 4px 0; }}
.pill {{ display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: 12.5px; font-weight: 600; }}
.up {{ background: rgba(224,36,94,.15); color: #E0245E; }} .down {{ background: rgba(34,197,94,.15); color: #22C55E; }}
.ok {{ background: rgba(34,197,94,.15); color: #22C55E; }} .warnp {{ background: rgba(245,194,66,.18); color: #C99A1D; }}
.tbl {{ width: 100%; border-collapse: collapse; font-size: 15px; border: none !important; }}
.stApp .tbl td, .stApp .tbl th, .stApp .tbl tr {{ border-left: none !important; border-right: none !important; border-top: none !important; background: transparent !important; }}
.tbl th {{ text-align: right; color: var(--muted); font-weight: 600; padding: 8px 6px; border-bottom: 1px solid var(--line); }}
.tbl td {{ text-align: right; padding: 9px 6px; border-bottom: 1px solid var(--line); }}
.tbl th:first-child, .tbl td:first-child {{ text-align: left; }}
.dot {{ display: inline-block; width: 11px; height: 11px; border-radius: 3px; margin-right: 8px; vertical-align: middle; }}
.stApp .note {{ color: var(--muted) !important; font-size: 14.5px; line-height: 1.8; margin-top: 12px; }}
.note b {{ color: var(--text); }}
.alert-row {{ padding: 12px 0; border-bottom: 1px solid var(--line); font-size: 15px; line-height: 1.5; }}
.alert-row:last-child {{ border-bottom: none; }}
.point {{ padding: 10px 0; border-bottom: 1px solid var(--line); font-size: 15.5px; line-height: 1.55; }}
.point:last-child {{ border-bottom: none; }}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ helpers
def t12(t: dt.time | str) -> str:
    if isinstance(t, str):
        h, m = map(int, t.split(":")[:2])
    else:
        h, m = t.hour, t.minute
    return f"{h % 12 or 12}{':%02d' % m if m else ''} {'AM' if h < 12 else 'PM'}"


def units(v: float) -> str:
    """Indian style: 4.7 lakh units / 12,345 units."""
    if v >= 1e7:
        return f"{v / 1e7:,.2f} crore units"
    if v >= 1e5:
        return f"{v / 1e5:,.1f} lakh units"
    return f"{v:,.0f} units"


def rupees(v: float) -> str:
    if v >= 1e7:
        return f"₹{v / 1e7:,.2f} crore"
    if v >= 1e5:
        return f"₹{v / 1e5:,.1f} lakh"
    return f"₹{v:,.0f}"


def header(title: str, sub: str):
    c1, c2 = st.columns([5, 1.3], vertical_alignment="center")
    c1.markdown(f'<p class="page-title">{title}</p><p class="page-sub">{sub}</p>', unsafe_allow_html=True)
    c2.toggle("🌙 Dark mode", value=dark, key="dark")
    st.write("")


def info_btn(key: str, where=None):
    with (where or st).popover(":material/info:", help="How is this calculated?"):
        st.markdown(INFO[key])


def read_aloud(text: str, key: str):
    """'Read this page' button at the top of every page (free browser voice)."""
    voice_player(text, f"page_{key}", "🔊 Read this page aloud", dark=dark)


def card_title(title: str, sub: str = "", info: str | None = None):
    html_ = f'<p class="card-title">{title}</p>' + (f'<p class="card-sub">{sub}</p>' if sub else "")
    if info:
        a, b = st.columns([14, 1], vertical_alignment="top")
        a.markdown(html_, unsafe_allow_html=True)
        info_btn(info, b)
    else:
        st.markdown(html_, unsafe_allow_html=True)


def plot(fig, height=300, legend=True):
    fig.update_layout(template="plotly_dark" if dark else "plotly_white", height=height,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter, Segoe UI, Arial", size=14, color=T["text"]),
                      margin=dict(l=4, r=4, t=10, b=4), showlegend=legend,
                      legend=dict(orientation="h", y=1.12, x=0, title=None, font=dict(size=13, color=T["text"])),
                      uniformtext=dict(minsize=11, mode="hide"))
    fig.update_xaxes(showgrid=False, linecolor=T["line"], title_font=dict(size=13, color=T["muted"]))
    fig.update_yaxes(gridcolor=T["grid"], zeroline=False, title_font=dict(size=13, color=T["muted"]))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def ask_box(sid: str, topic: str, context: dict, suggestions: list[str]):
    desc, keys = PAGE_DESC[sid]
    context = {"page_description": desc, "how_things_are_calculated": [INFO[k] for k in keys], **context}
    with st.container(border=True):
        card_title("💬 Have a question?", f"Ask anything about {topic} in your own words")
        history = st.container()
        cols = st.columns(len(suggestions))
        clicked = None
        for i, (c, s) in enumerate(zip(cols, suggestions)):
            if c.button(s, key=f"sug_{sid}_{i}", width="stretch"):
                clicked = s
        q = st.chat_input("Type your question…", key=f"chat_{sid}") or clicked
        log = st.session_state.setdefault(f"log_{sid}", [])
        if q:
            with st.spinner("Thinking…"):
                log.append({"role": "user", "text": q})
                log.append({"role": "assistant", **ask(q, topic, context, cfg["provider"])})
        with history:
            for m in log[-4:]:
                with st.chat_message(m["role"], avatar="🙂" if m["role"] == "user" else "⚡"):
                    st.markdown(m["text"] if m["role"] == "user" else m["answer"])
                    if m["role"] == "assistant":
                        st.caption(f"Answered by {m['source']}" + ("" if m["verified"] else
                                   " · ⚠️ some numbers could not be confirmed"))


# ------------------------------------------------------------------ data + analysis (cached)
@st.cache_data(show_spinner=False)
def load(data: bytes | None):
    return load_and_clean(SAMPLE if data is None else io.BytesIO(data))


@st.cache_data(show_spinner=False)
def analyse(df, shifts_json, interval, label_end):
    sh = json.loads(shifts_json)
    a = assign_shifts(df, sh, interval, label_end)
    return a, compute_shift_stats(a, sh, interval)


@st.cache_data(show_spinner=False)
def find_alerts(assigned, z, interval):
    scored = score_readings(assigned)
    return scored, detect_alerts(scored, z, 5.0, interval)


# ------------------------------------------------------------------ sidebar menu
with st.sidebar:
    st.markdown('<p style="font-size:24px;font-weight:800;margin:6px 0 2px">⚡ ShiftWatt</p>'
                '<p class="page-sub" style="margin-bottom:18px">Electricity by work shift, made simple</p>',
                unsafe_allow_html=True)
    page = st.radio("Menu", PAGES, label_visibility="collapsed", key="page")

# ------------------------------------------------------------------ compute everything once
try:
    df, report = load(cfg["upload"])
except Exception as e:  # noqa: BLE001
    st.error(f"We could not read your file: {e}. Go to Settings and choose another file.")
    cfg["upload"] = None
    st.stop()

def cfg_shifts() -> list[dict]:
    return [{"name": x["name"].strip() or f"Shift {i + 1}", "start": x["start"].strftime("%H:%M"),
             "end": x["end"].strftime("%H:%M")} for i, x in enumerate(cfg["shifts"])]


def clear_shift_widgets():
    for k in list(st.session_state):
        if k[:2] in ("sn", "ss", "se") and k[2:].isdigit():
            del st.session_state[k]


def split_evenly(count: int, start: dt.time):
    """Split the 24-hour day into `count` equal shifts starting at `start`, keeping existing names."""
    names = [x["name"] for x in cfg["shifts"]]
    m0 = start.hour * 60 + start.minute
    cfg["shifts"] = []
    for k in range(count):
        a0, b0 = (m0 + k * 1440 // count) % 1440, (m0 + (k + 1) * 1440 // count) % 1440
        cfg["shifts"].append({"name": names[k] if k < len(names) else f"Shift {chr(65 + k)}",
                              "start": dt.time(a0 // 60, a0 % 60), "end": dt.time(b0 // 60, b0 % 60)})
    clear_shift_widgets()


shifts = cfg_shifts()
interval = report["interval_minutes"]
assigned, facts = analyse(df, json.dumps(shifts), interval, cfg["label_end"])
scored, events = find_alerts(assigned, PICKY[cfg["picky"]], interval)
facts = {**facts, "alerts_summary": alert_summary(events, shifts)}
S = pd.DataFrame(facts["shifts"])
S = S[S["readings"] > 0].reset_index(drop=True)
C = facts["comparisons"]
color = {s["name"]: SHIFT_COLORS[i % len(SHIFT_COLORS)] for i, s in enumerate(shifts)}
label = {s["name"]: f"{s['name']} ({t12(s['start'])}–{t12(s['end'])})" for s in shifts}
total = float(S["total_kwh"].sum()) if len(S) else 0.0
_empty = [x["name"] for x in shifts if x["name"] not in set(S["name"])]
if (_empty or any("overlap" in w for w in validate_shifts(shifts))) and page != PAGES[8]:
    st.warning("⚠️ Some of your shift times overlap" + (f", so {', '.join(_empty)} got no readings" if _empty else "")
               + ". Overlapping time is counted for the shift listed first. Fix this in ⚙️ Settings.")
if S.empty:
    header("⚠️ Nothing to show", "No electricity readings fall inside your shift times.")
    st.info("Open ⚙️ Settings and check your shift times.")
    st.stop()

# ================================================================== HOME
if page == PAGES[0]:
    hi, lo = C.get("highest_total"), C.get("lowest_total")
    header("Your factory's electricity", f"{pd.Timestamp(report['start']):%d %b %Y} to {pd.Timestamp(report['end']):%d %b %Y}"
           " · shift times set by you in Settings")
    _hr = S.loc[S["energy_per_operating_hour"].idxmax()]
    read_aloud(f"Your factory used {units(total)} of electricity in this period. "
               + (f"{hi} used the most: {S.set_index('name').loc[hi, 'total_kwh'] / total * 100:.0f} percent of all "
                  f"electricity. {lo} used the least. " if hi and lo else "")
               + f"For each hour of work, {_hr['name']} used the most, about {_hr['energy_per_operating_hour']:.0f} units. "
               f"The estimated bill is {rupees(total * cfg['price'])}. We found {len(events)} times of unusual usage. "
               "Shift times are your own settings.", "home")

    daily = daily_shift_totals(assigned)
    daily["date"] = pd.to_datetime(daily["date"])
    days_in = daily.groupby(pd.Grouper(key="date", freq="MS"))["date"].nunique()
    monthly = daily.groupby([pd.Grouper(key="date", freq="MS"), "shift"])["total_kwh"].sum().reset_index()
    monthly = monthly[monthly["date"].isin(days_in[days_in >= 7].index)]

    with st.container(border=True):
        card_title("Summary", "How much electricity each shift used", "summary")
        cols = st.columns(len(S))
        for col, (_, r) in zip(cols, S.iterrows()):
            n = r["name"]
            tag = ('<span class="pill up">▲ uses the most</span>' if n == hi else
                   '<span class="pill down">▼ uses the least</span>' if n == lo else
                   '<span class="pill" style="visibility:hidden">·</span>')
            col.markdown(f'<div class="mini-head"><span class="dot" style="background:{color[n]}"></span>{html.escape(n)}</div>'
                         f'<div class="mini-label">{t12(shifts[[s["name"] for s in shifts].index(n)]["start"])} – '
                         f'{t12(shifts[[s["name"] for s in shifts].index(n)]["end"])}</div>'
                         f'<div class="big">{units(r["total_kwh"])}</div>{tag}', unsafe_allow_html=True)
            m = monthly[monthly["shift"] == n]
            if len(m):
                f = go.Figure(go.Scatter(x=m["date"], y=m["total_kwh"], mode="lines", line=dict(color=color[n], width=3,
                                         shape="spline"), fill="tozeroy", fillcolor=color[n] + "22",
                                         hovertemplate="%{x|%b}: %{y:,.0f} units<extra></extra>"))
                top = m.loc[m["total_kwh"].idxmax()]
                f.add_trace(go.Scatter(x=[top["date"]], y=[top["total_kwh"]], mode="markers",
                                       marker=dict(color="#fff", size=9, line=dict(color=color[n], width=3)),
                                       hoverinfo="skip"))
                f.update_xaxes(visible=False)
                f.update_yaxes(visible=False)
                with col:
                    plot(f, 110, legend=False)
                    st.caption(f"Busiest month: {top['date']:%B}")

    left, right = st.columns([1.35, 1])
    with left, st.container(border=True):
        card_title("Shift summary", "Units of electricity used (1 unit = 1 kWh, the same unit as on your bill)", "table")
        rows = "".join(
            f'<tr><td><span class="dot" style="background:{color[r["name"]]}"></span>{html.escape(r["name"])}</td>'
            f'<td>{r["total_kwh"]:,.0f}</td><td>{r["total_kwh"] / total * 100:.0f}%</td>'
            f'<td>{r["energy_per_operating_hour"]:,.0f}</td></tr>' for _, r in S.iterrows())
        st.markdown(f'<table class="tbl"><tr><th>Shift</th><th>Units used</th><th>Share</th><th>Units per hour of work</th>'
                    f'</tr>{rows}<tr><td><b>Total</b></td><td><b>{total:,.0f}</b></td><td><b>100%</b></td><td></td></tr>'
                    f'</table>', unsafe_allow_html=True)
        hr = S.loc[S["energy_per_operating_hour"].idxmax()]
        wd, we = S["weekday_avg_daily_kwh"].sum(), S["weekend_avg_daily_kwh"].sum()
        notes = [f"<b>{html.escape(hi)}</b> used <b>{S.set_index('name').loc[hi, 'total_kwh'] / total * 100:.0f}%</b> "
                 f"of all electricity." if hi else "",
                 f"For each hour of work, <b>{html.escape(hr['name'])}</b> used the most: about "
                 f"<b>{hr['energy_per_operating_hour']:,.0f} units</b> every hour.",
                 f"Weekends use <b>{(1 - we / wd) * 100:.0f}% less</b> electricity than weekdays." if wd and we < wd else ""]
        st.markdown('<div class="note">' + "<br>".join(f"▸ {x}" for x in notes if x) + "</div>", unsafe_allow_html=True)

    with right, st.container(border=True):
        card_title("Who used the electricity?", "Each shift's share of the total", "donut")
        f = go.Figure(go.Pie(labels=S["name"], values=S["total_kwh"], hole=.62, sort=False,
                             marker=dict(colors=[color[n] for n in S["name"]], line=dict(color=T["panel"], width=3)),
                             textinfo="percent", textfont=dict(size=14, color="#fff"),
                             hovertemplate="%{label}: %{value:,.0f} units<extra></extra>"))
        f.add_annotation(text=f"<b>{units(total).replace(' units', '')}</b><br><span style='font-size:13px'>units in total</span>",
                         showarrow=False, font=dict(size=20, color=T["text"]))
        plot(f, 290)

    c1, c2, c3 = st.columns(3)
    for col, (icon, ttl, val, sub, ik) in zip((c1, c2, c3), [
        ("💰", "Estimated electricity bill", rupees(total * cfg["price"]), f"at ₹{cfg['price']:g} per unit (change in Settings)", "bill"),
        ("🔔", "Unusual usage found", f"{len(events)} times", "see 🔔 Unusual usage in the menu", "unusual"),
        ("🌱", "Carbon pollution", f"{S['total_co2'].sum():,.0f} tonnes" if S["total_co2"].notna().any() else "–",
         "from producing this electricity", "co2tile"),
    ]):
        with col, st.container(border=True):
            a, b = st.columns([6, 1])
            a.markdown(f'<div class="mini-label">{icon} {ttl}</div><div class="big">{val}</div>'
                       f'<div class="mini-label">{sub}</div>', unsafe_allow_html=True)
            info_btn(ik, b)

    ask_box("home", "this summary",
            {"shifts": S[["name", "total_kwh", "energy_per_operating_hour", "weekday_avg_daily_kwh",
                          "weekend_avg_daily_kwh", "load_type_mix"]].to_dict("records"),
             "comparisons": C, "shift_times_set_by_user": shifts, "price_per_unit_rupees": cfg["price"]},
            ["What is a unit?", "Which shift uses the most?", "Why does one shift use more?"])

# ================================================================== UNUSUAL USAGE
elif page == PAGES[1]:
    header("Unusual usage", "Times when electricity use was far from normal")
    if events.empty:
        read_aloud("No unusual usage was found at the current setting.", "alerts")
    else:
        _top = events.iloc[0]
        read_aloud(f"With the setting {cfg['picky']}, usage was far above normal "
                   f"{int((events['direction'] == 'HIGH').sum())} times and far below normal "
                   f"{int((events['direction'] == 'LOW').sum())} times. The biggest case was in {_top['shift']} on "
                   f"{_top['start']:%A %d %B}: {_top['worst_kwh']:.0f} units in 15 minutes, when normal is about "
                   f"{_top['typical_kwh']:.0f}.", "alerts")
    with st.container(border=True):
        card_title("How picky should we be?",
                   "We compare every 15 minutes with the same day and time in other weeks (for example, other "
                   "Thursday nights at 11 PM). If it is far off, we show it here.", "picky")
        choice = st.radio("Show me", list(PICKY), index=list(PICKY).index(cfg["picky"]), horizontal=True,
                          label_visibility="collapsed", key="picky_w")
        if choice != cfg["picky"]:
            cfg["picky"] = choice
            st.rerun()
        st.caption({"Only big problems": "Only the most extreme cases. Fewest results.",
                    "Big and medium problems": "Recommended. Big cases plus clearly unusual ones.",
                    "Everything unusual": "Shows even smaller differences. Most results."}[cfg["picky"]])

    if events.empty:
        st.success("Nothing unusual found. Usage stayed close to normal.")
    else:
        hi_n, lo_n = int((events["direction"] == "HIGH").sum()), int((events["direction"] == "LOW").sum())
        c1, c2 = st.columns(2)
        with c1, st.container(border=True):
            st.markdown(f'<div class="mini-label">🔺 Used far MORE than normal</div><div class="big">{hi_n} times</div>'
                        '<div class="mini-label">could mean waste, a machine left on, or extra work</div>',
                        unsafe_allow_html=True)
        with c2, st.container(border=True):
            st.markdown(f'<div class="mini-label">🔻 Used far LESS than normal</div><div class="big">{lo_n} times</div>'
                        '<div class="mini-label">could mean a stoppage, breakdown or holiday</div>', unsafe_allow_html=True)

        def say(e) -> str:
            s0, s1 = e["start"], e["end"]
            when = (f"{s0:%a %d %b}, {t12(s0.time())} to {t12(s1.time())}" if s0.date() == s1.date()
                    else f"{s0:%a %d %b} {t12(s0.time())} to {s1:%a} {t12(s1.time())}")
            if e["direction"] == "HIGH":
                times = e["worst_kwh"] / e["typical_kwh"] if e["typical_kwh"] else None
                how = (f"used up to <b>{e['worst_kwh']:.0f} units</b> in 15 minutes, when normal is about "
                       f"{e['typical_kwh']:.0f}" + (f": about <b>{times:.0f} times</b> normal" if times and times >= 2 else ""))
            else:
                how = (f"dropped to <b>{e['worst_kwh']:.0f} units</b> in 15 minutes, when normal is about "
                       f"{e['typical_kwh']:.0f}: <b>{(1 - e['worst_kwh'] / e['typical_kwh']) * 100:.0f}% less</b>")
            icon = "🔺" if e["direction"] == "HIGH" else "🔻"
            return f"{icon} <b>{html.escape(e['shift'])}</b> · {when} · {how}."

        with st.container(border=True):
            card_title("The biggest ones", "Most unusual first", "biggest")
            top5 = events.head(5)
            st.markdown("".join(f'<div class="alert-row">{say(e)}</div>' for _, e in top5.iterrows()),
                        unsafe_allow_html=True)
            plain_txt = [say(e).replace("<b>", "").replace("</b>", "").replace("🔺 ", "").replace("🔻 ", "")
                         for _, e in top5.iterrows()]
            voice_player("The biggest unusual usage. " + " ".join(plain_txt), "alerts", "🔊 Read these aloud", dark=dark)

        with st.container(border=True):
            card_title("Look at one day", "Blue = what happened that day · Grey dashed = a normal day like it", "day")
            dates = sorted(events["start"].dt.date.unique())
            day = st.selectbox("Choose a day", dates, index=dates.index(events.iloc[0]["start"].date()),
                               format_func=lambda d: d.strftime("%A, %d %B %Y"))
            dv = day_view(scored, day)
            f = go.Figure()
            f.add_trace(go.Scatter(x=dv["interval_start"], y=dv["expected_kwh"], name=f"A normal {day:%A}",
                                   line=dict(color=T["muted"], dash="dash", width=2)))
            f.add_trace(go.Scatter(x=dv["interval_start"], y=dv["energy_kwh"], name="That day",
                                   line=dict(color="#1BA3B8", width=3)))
            f.update_yaxes(title="units every 15 minutes")
            f.update_xaxes(tickformat="%I %p")
            plot(f, 300)

    ask_box("alerts", "unusual usage",
            {"shifts": [{"name": n, **v} for n, v in facts["alerts_summary"].items()], "total": len(events),
             "biggest_cases": events.head(6)["message"].tolist() if len(events) else [], "picky_setting": cfg["picky"]},
            ["What does unusual mean?", "Is 'more than normal' bad?", "Why did this happen?"])

# ================================================================== LIVE MONITOR (simulation)
elif page == PAGES[2]:
    header("Live monitor", "A simulation: we replay a recorded day as if readings were arriving live, and alert you "
                           "the moment usage crosses your limit")
    _log = [m.split(" ", 1)[1] for m in st.session_state.get("sim", {}).get("log", [])[:3] if " " in m]
    read_aloud("This page replays a recorded day as if the meter readings were arriving live. "
               f"You get an alert when usage goes above {cfg.get('limit', 90)} units in 15 minutes. Press Start to begin. "
               + ("Latest alerts: " + " ".join(_log) if _log else ""), "live")
    sim = st.session_state.setdefault("sim", {"running": False, "i": 0, "day": None, "above": False, "far": False,
                                               "log": [], "new": []})
    days = sorted(scored["interval_start"].dt.date.unique())
    default_day = events.iloc[0]["start"].date() if len(events) else days[len(days) // 2]
    with st.container(border=True):
        card_title("1. Choose what to watch", "Pick a day to replay and the limit that should trigger an alert", "live")
        c1, c2, c3 = st.columns(3)
        day = c1.selectbox("Day to replay", days, index=days.index(cfg.get("sim_day", default_day)),
                           format_func=lambda d: d.strftime("%a %d %b %Y"))
        cfg["limit"] = c2.number_input("Alert me when usage goes above (units in 15 min)", 1, 1000,
                                       int(cfg.get("limit", 90)), 5)
        speed = c3.select_slider("Replay speed", ["Slow", "Normal", "Fast", "Very fast"], value=cfg.get("speed", "Fast"))
        cfg["speed"] = speed
        cfg["smart"] = st.checkbox("Also alert when usage is far from normal for that time of the week "
                                   "(uses the 'How picky' setting on the Unusual usage page)", cfg.get("smart", True))
        if day != cfg.get("sim_day"):
            cfg["sim_day"] = day
            sim.update(running=False, i=0, above=False, far=False, log=[], new=[])
        b1, b2, b3 = st.columns(3)
        n_day = int((scored["interval_start"].dt.date == cfg["sim_day"]).sum())
        done = sim["i"] >= n_day
        if b1.button("▶ Start" if sim["i"] == 0 else "▶ Replay again" if done else "▶ Resume", type="primary",
                     width="stretch", disabled=sim["running"]):
            if done:
                sim.update(i=0, above=False, far=False, log=[], new=[])
            sim["running"] = True
            st.rerun()
        if b2.button("⏸ Pause", width="stretch", disabled=not sim["running"]):
            sim["running"] = False
            st.rerun()
        if b3.button("↺ Restart", width="stretch"):
            sim.update(running=False, i=0, above=False, far=False, log=[], new=[])
            st.rerun()
        n1, n2 = st.columns([2, 3])
        with n1:
            enable_button(dark)
        n2.caption("Pop-ups always appear on screen. Desktop notifications need your permission once. "
                   + ("📱 Phone alerts via Telegram are ON." if cfg.get("tg_token") and cfg.get("tg_chat")
                      else "📱 Phone alerts: set up Telegram in ⚙️ Settings (optional)."))

    STEP = {"Slow": 1, "Normal": 2, "Fast": 4, "Very fast": 8}[speed]
    zthr = PICKY[cfg["picky"]]

    @st.fragment(run_every=1.0 if sim["running"] else None)
    def live_view():
        dv = day_view(scored, cfg["sim_day"]).reset_index(drop=True)
        if sim["running"]:
            start_i, sim["i"] = sim["i"], min(len(dv), sim["i"] + STEP)
            for _, r in dv.iloc[start_i:sim["i"]].iterrows():
                v, e, z = r["energy_kwh"], r["expected_kwh"], r["z"]
                when = t12(r["timestamp"].time())
                over = v > cfg["limit"]
                far = cfg["smart"] and abs(z) >= zthr and abs(v - e) >= 5
                msgs = []
                if over and not sim["above"]:
                    msgs.append(f"🚨 {r['shift']}: {v:.0f} units at {when}, above your limit of {cfg['limit']}")
                if far and not sim["far"]:
                    msgs.append(f"⚠️ {r['shift']}: {v:.0f} units at {when}, far {'above' if v > e else 'below'} "
                                f"normal (normally about {e:.0f})")
                sim["above"], sim["far"] = over, far
                for m in msgs:
                    sim["log"].insert(0, m)
                    sim["new"].append(m)
            if sim["i"] >= len(dv):
                sim["running"] = False
                sim["log"].insert(0, "✅ Replay finished.")
        for k, m in enumerate(sim["new"]):  # pop-ups + desktop + phone for alerts that just happened
            st.toast(m, icon="🚨")
            desktop_alert("ShiftWatt alert", m.split(" ", 1)[1], f"{cfg['sim_day']}-{len(sim['log'])}-{k}")
            if cfg.get("tg_token") and cfg.get("tg_chat"):
                send_telegram(cfg["tg_token"], cfg["tg_chat"], "ShiftWatt: " + m.split(" ", 1)[1])
        sim["new"] = []

        cur = dv.iloc[:sim["i"]]
        last = cur.iloc[-1] if len(cur) else None
        with st.container(border=True):
            k1, k2, k3, k4 = st.columns(4)
            status = ("🔴 LIVE" if sim["running"] else "⏸ Paused" if 0 < sim["i"] < len(dv) else
                      "✅ Finished" if sim["i"] >= len(dv) else "⏹ Not started")
            k1.markdown(f'<div class="mini-label">Status</div><div class="big">{status}</div>', unsafe_allow_html=True)
            k2.markdown(f'<div class="mini-label">Time now</div><div class="big">'
                        f'{t12(last["timestamp"].time()) if last is not None else "–"}</div>'
                        f'<div class="mini-label">{last["shift"] if last is not None else ""}</div>', unsafe_allow_html=True)
            col = "#E0245E" if last is not None and last["energy_kwh"] > cfg["limit"] else T["text"]
            k3.markdown(f'<div class="mini-label">Latest reading</div><div class="big" style="color:{col}">'
                        f'{last["energy_kwh"]:.0f} units</div><div class="mini-label">in the last 15 minutes</div>'
                        if last is not None else '<div class="mini-label">Latest reading</div><div class="big">–</div>',
                        unsafe_allow_html=True)
            k4.markdown(f'<div class="mini-label">Alerts so far</div><div class="big">'
                        f'{sum(1 for m in sim["log"] if not m.startswith("✅"))}</div>', unsafe_allow_html=True)
            f = go.Figure()
            f.add_trace(go.Scatter(x=dv["interval_start"], y=dv["expected_kwh"], name="Normal for this day of the week",
                                   line=dict(color=T["muted"], dash="dash", width=2)))
            f.add_trace(go.Scatter(x=cur["interval_start"], y=cur["energy_kwh"], name="Live reading",
                                   line=dict(color="#1BA3B8", width=3)))
            hot = cur[cur["energy_kwh"] > cfg["limit"]]
            f.add_trace(go.Scatter(x=hot["interval_start"], y=hot["energy_kwh"], mode="markers", name="Above your limit",
                                   marker=dict(color="#E0245E", size=9)))
            f.add_hline(y=cfg["limit"], line=dict(color="#E0245E", dash="dot"), annotation_text=f"Your limit: {cfg['limit']}",
                        annotation_position="top left", annotation_font=dict(color="#E0245E"))
            f.update_xaxes(range=[dv["interval_start"].min(), dv["interval_start"].max()], tickformat="%I %p")
            f.update_yaxes(title="units every 15 minutes", rangemode="tozero")
            plot(f, 340)
        with st.container(border=True):
            card_title("Alert log", "Newest first")
            st.markdown("".join(f'<div class="alert-row">{html.escape(m)}</div>' for m in sim["log"][:12])
                        or '<div class="mini-label">No alerts yet. Press ▶ Start.</div>', unsafe_allow_html=True)
        if not sim["running"] and sim["i"] >= len(dv) and sim.get("_was_running"):
            sim["_was_running"] = False
            st.rerun()
        sim["_was_running"] = sim["running"]

    live_view()
    ask_box("live", "the live monitor",
            {"limit_units_per_15min": cfg["limit"], "far_from_normal_alerts_on": cfg["smart"], "picky": cfg["picky"],
             "alerts_so_far": sim["log"][:10]},
            ["How does the live monitor work?", "Is this real-time data?", "What happens with a real meter?"])

# ================================================================== DAILY PATTERN
elif page == PAGES[3]:
    header("Daily pattern", "When during the day is electricity used?")
    t = assigned["interval_start"]
    prof = assigned.assign(hour=t.dt.hour + t.dt.minute / 60,
                           day=t.dt.dayofweek.map(lambda d: "Weekend (Sat–Sun)" if d >= 5 else "Weekday (Mon–Fri)"))
    prof = prof.groupby(["day", "hour"])["energy_kwh"].mean().reset_index()
    wk = prof[prof["day"].str.startswith("Weekday")]
    busy, quiet = wk.loc[wk["energy_kwh"].idxmax()], wk.loc[wk["energy_kwh"].idxmin()]
    to_t = lambda h: t12(dt.time(int(h), int(round(h % 1 * 60)) % 60))  # noqa: E731
    _wk_hi = S.loc[(S["weekday_avg_daily_kwh"] - S["weekend_avg_daily_kwh"]).idxmax()]
    read_aloud(f"On a normal weekday, electricity use is highest around {to_t(busy['hour'])} and lowest around "
               f"{to_t(quiet['hour'])}. {_wk_hi['name']} uses about {_wk_hi['weekday_avg_daily_kwh']:.0f} units on a "
               f"weekday and {_wk_hi['weekend_avg_daily_kwh']:.0f} units on a weekend day.", "pattern")
    with st.container(border=True):
        card_title("A normal day", f"On weekdays, usage is highest around <b>{to_t(busy['hour'])}</b> and lowest around "
                                   f"<b>{to_t(quiet['hour'])}</b>. The coloured areas are your shifts.", "normalday")
        f = px.line(prof, x="hour", y="energy_kwh", color="day", line_shape="spline",
                    color_discrete_map={"Weekday (Mon–Fri)": "#1BA3B8", "Weekend (Sat–Sun)": "#F5C242"},
                    labels={"hour": "", "energy_kwh": "units every 15 minutes", "day": ""})
        f.update_traces(line=dict(width=3))
        for s in shifts:
            a0, b0 = _minutes(s["start"]) / 60, _minutes(s["end"]) / 60
            for x0, x1 in ([(a0, b0)] if a0 < b0 else [(a0, 24), (0, b0)]):
                f.add_vrect(x0=x0, x1=x1, fillcolor=color[s["name"]], opacity=0.10, line_width=0)
            f.add_annotation(x=a0 + 0.2, y=1.0, yref="paper", text=f"<b>{s['name']}</b>", showarrow=False,
                             xanchor="left", yanchor="top", font=dict(color=color[s["name"]], size=13))
        f.update_xaxes(tickvals=list(range(0, 25)), ticktext=[t12(dt.time(h % 24)).replace(" ", "") for h in range(25)],
                       range=[0, 24], tickangle=-45, tickfont=dict(size=11))
        plot(f, 380)

    with st.container(border=True):
        card_title("Weekdays vs weekends", "Average units used in one day, per shift", "wkwe")
        ww = S.melt(id_vars="name", value_vars=["weekday_avg_daily_kwh", "weekend_avg_daily_kwh"], var_name="d",
                    value_name="u")
        ww["d"] = ww["d"].map({"weekday_avg_daily_kwh": "Weekday", "weekend_avg_daily_kwh": "Weekend"})
        f = px.bar(ww, x="name", y="u", color="d", barmode="group", text_auto=",.0f",
                   color_discrete_map={"Weekday": "#1BA3B8", "Weekend": "#F5C242"}, labels={"name": "", "u": "units", "d": ""})
        f.update_traces(textposition="outside", cliponaxis=False)
        plot(f, 300)

    ask_box("pattern", "the daily pattern",
            {"weekday_busiest_time": to_t(busy["hour"]), "weekday_quietest_time": to_t(quiet["hour"]),
             "shifts": S[["name", "weekday_avg_daily_kwh", "weekend_avg_daily_kwh"]].to_dict("records"),
             "shift_times_set_by_user": shifts},
            ["How do I read this chart?", "Why is the weekend lower?", "Which shift is busiest?"])

# ================================================================== POLLUTION & EFFICIENCY
elif page == PAGES[4]:
    header("Pollution & efficiency", "Carbon pollution, how well electricity is used, and how busy each shift was")
    _parts = []
    if S["total_co2"].notna().any():
        _c = S.loc[S["total_co2"].idxmax()]
        _parts.append(f"{_c['name']} caused the most carbon pollution: {_c['total_co2']:.0f} tonnes.")
    if S["avg_power_factor"].notna().any():
        _p = S.loc[S["avg_power_factor"].idxmin()]
        _parts.append(f"{_p['name']} had the lowest efficiency score: {_p['avg_power_factor']:.0f} out of 100. "
                      "Higher is better.")
    read_aloud(" ".join(_parts) or "This file has no pollution or efficiency information.", "co2")
    has_co2, has_pf, has_mix = (S["total_co2"].notna().any(), S["avg_power_factor"].notna().any(),
                                S["load_type_mix"].notna().any())
    if not (has_co2 or has_pf or has_mix):
        st.info("Your file does not include this information.")
    c1, c2 = st.columns(2)
    if has_co2:
        with c1, st.container(border=True):
            cm = S.loc[S["total_co2"].idxmax()]
            card_title("Carbon pollution", f"Tonnes of CO₂. <b>{html.escape(cm['name'])}</b> caused the most.", "carbon")
            f = px.bar(S, x="name", y="total_co2", color="name", color_discrete_map=color, text_auto=",.0f",
                       labels={"name": "", "total_co2": "tonnes"})
            f.update_traces(textposition="outside", cliponaxis=False)
            plot(f, 290, legend=False)
    if has_pf:
        with c2, st.container(border=True):
            pm = S.loc[S["avg_power_factor"].idxmin()]
            card_title("Efficiency score (out of 100)",
                       f"Higher is better: less electricity wasted. <b>{html.escape(pm['name'])}</b> scored lowest.", "eff")
            f = px.bar(S, x="name", y="avg_power_factor", color="name", color_discrete_map=color, text_auto=".0f",
                       labels={"name": "", "avg_power_factor": "score"})
            f.update_traces(textposition="outside", cliponaxis=False)
            f.update_yaxes(range=[0, 110])
            plot(f, 290, legend=False)
    if has_mix:
        with st.container(border=True):
            card_title("How busy was each shift?", "Share of each shift's time that the factory was quiet, normal or very busy", "busy")
            mix = pd.DataFrame([{"Shift": r["name"], "How busy": {"Light_Load": "Quiet", "Medium_Load": "Normal",
                                                                    "Maximum_Load": "Very busy"}.get(k, k), "%": v}
                                for _, r in S.iterrows() if r["load_type_mix"] for k, v in r["load_type_mix"].items()])
            f = px.bar(mix, x="%", y="Shift", color="How busy", orientation="h",
                       category_orders={"How busy": ["Quiet", "Normal", "Very busy"]},
                       color_discrete_map={"Quiet": "#1BA3B8", "Normal": "#F5C242", "Very busy": "#E0245E"},
                       labels={"%": "% of the shift's time", "Shift": ""})
            f.update_traces(texttemplate="%{x:.0f}%", textposition="inside")
            plot(f, 240)
    ask_box("co2", "pollution and efficiency",
            {"shifts": S[["name", "total_co2", "avg_power_factor", "load_type_mix"]].to_dict("records")},
            ["What is the efficiency score?", "What is carbon pollution?", "What does 'very busy' mean?"])

# ================================================================== WRITTEN SUMMARY
elif page == PAGES[5]:
    header("Written summary", "A short report in plain words, with every number double-checked")
    if st.session_state.get("notes"):
        _n = st.session_state["notes"]
        read_aloud(" ".join([_n["summary"]] + [f"{o['title']}. {o['text']}" for o in _n["observations"]]), "brief")
    else:
        read_aloud("Press Write my summary to get a short report in plain words. Every number in it is double-checked. "
                   "You can then listen to it in nine languages.", "brief")
    used = resolve_provider(cfg["provider"])
    fkey = json.dumps(facts, sort_keys=True, default=str) + cfg["provider"]
    with st.container(border=True):
        a_, b_ = st.columns([14, 1])
        go_btn = a_.button("✨ Write my summary", type="primary")
        info_btn("summarytext", b_)
        st.caption("Written by " + {"gemini": "Google Gemini (free)", "ollama": f"Ollama on this computer ({ollama_model()})"}
                   .get(used, "the built-in writer (add a free AI in Settings for richer wording)"))
        if go_btn and st.session_state.get("notes_key") != fkey:
            with st.spinner("Writing and checking…"):
                st.session_state["notes"] = generate_notes(facts, cfg["provider"])
                st.session_state["notes_key"] = fkey
        if st.session_state.get("notes_key") == fkey:
            notes = st.session_state["notes"]
            v = notes["verification"]
            st.markdown((f'<span class="pill ok">✓ All {v["checked"]} numbers double-checked against the data</span>'
                         if v["all_verified"] else
                         f'<span class="pill warnp">⚠ {len(v["mismatches"])} number(s) could not be confirmed</span>'),
                        unsafe_allow_html=True)
            st.markdown(f'<p style="font-size:17px;line-height:1.6;margin:14px 0 6px">{html.escape(notes["summary"])}</p>',
                        unsafe_allow_html=True)
            st.markdown("".join(f'<div class="point">{"✓" if o.get("verified") else "⚠"} <b>{html.escape(o["title"])}:</b> '
                                f'{html.escape(o["text"])}</div>' for o in notes["observations"]), unsafe_allow_html=True)
            st.caption(notes["assumption_statement"])

    if st.session_state.get("notes_key") == fkey:
        notes = st.session_state["notes"]
        with st.container(border=True):
            card_title("🔊 Listen", "Hear the summary in your language")
            l1, l2 = st.columns([3, 1])
            lang_label = l1.selectbox("Language", list(LANGS), key="lang", label_visibility="collapsed")
            code, lname = LANGS[lang_label]
            src_text = brief_text(notes)
            if code == "en":
                voice_player(src_text, "brief_en", "▶ Listen in English", dark=dark)
            else:
                tkey = f"{code}:{hash(src_text)}"
                if l2.button("▶ Listen", width="stretch", type="primary"):
                    with st.spinner(f"Translating to {lname}…"):
                        tr, how = translate(src_text, code, lname)
                        st.session_state["tr"] = {"key": tkey, "text": tr,
                                                  "audio": tts_mp3(speech_ready(tr, code), code) if tr else None}
                trs = st.session_state.get("tr")
                if trs and trs["key"] == tkey:
                    if trs["text"] is None:
                        st.warning("This language needs internet and a free AI key (Settings). Hindi and Kannada work "
                                   "offline with the sample data.")
                    else:
                        tv = verify_translation(src_text, trs["text"], [s["name"] for s in shifts])
                        st.markdown(f'<span class="pill {"ok" if tv["ok"] else "warnp"}">'
                                    f'{"✓ All numbers match the English version" if tv["ok"] else "⚠ Some numbers changed in translation – do not rely on it"}'
                                    '</span>', unsafe_allow_html=True)
                        if trs["audio"]:
                            st.audio(trs["audio"], format="audio/mp3")
                        else:
                            voice_player(speech_ready(trs["text"], code), f"tr_{code}", f"▶ Listen in {lname}",
                                         lang=code, dark=dark)
                        st.markdown("".join(f'<div class="point">{html.escape(x)}</div>' for x in trs["text"].splitlines()),
                                    unsafe_allow_html=True)
        with st.expander("💾 Save a copy"):
            d1, d2 = st.columns(2)
            d1.download_button("Download summary", notes_to_markdown(notes, facts), "shift_summary.md", width="stretch")
            d2.download_button("Download the numbers", json.dumps(facts, indent=2, default=str), "shift_numbers.json",
                               width="stretch")
        ctx = {"summary": notes["summary"], "points": [o["text"] for o in notes["observations"]],
               "numbers_checked": notes["verification"]["checked"]}
    else:
        ctx = {"note": "No summary written yet."}
    ask_box("brief", "the written summary", ctx,
            ["How is this different from ChatGPT?", "What does double-checked mean?", "Summarise in one line"])

# ================================================================== DATA CHECK
elif page == PAGES[6]:
    r = report
    good = r["coverage_pct"] >= 95 and not r["warnings"]
    header("Data check", "Can these results be trusted?")
    read_aloud(("Your data is complete. " if good else "Your data has some gaps. ")
               + f"There are {r['rows']:,} meter readings, one every {r['interval_minutes']} minutes, covering "
               f"{r['coverage_pct']:.0f} percent of the expected time. Missing readings are never filled in.", "data")
    with st.container(border=True):
        st.markdown(f'<div class="big">{"✅ Your data is complete" if good else "⚠️ Your data has some gaps"}</div>'
                    f'<p class="page-sub">{"Nothing is missing, so the results are reliable." if good else "Results are still shown, but keep the gaps in mind."}'
                    " Missing readings are never filled in or guessed.</p>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for col, (ttl, val, sub), ik in zip((c1, c2, c3), [
        ("Meter readings", f"{r['rows']:,}", f"{pd.Timestamp(r['start']):%d %b %Y} – {pd.Timestamp(r['end']):%d %b %Y}"),
        ("A reading every", f"{r['interval_minutes']} minutes", "found automatically"),
        ("Data complete", f"{r['coverage_pct']:.0f}%", f"{r['missing_intervals']:,} readings missing"),
    ], ["readings", "every", "complete"]):
        with col, st.container(border=True):
            a, b = st.columns([6, 1])
            a.markdown(f'<div class="mini-label">{ttl}</div><div class="big">{val}</div><div class="mini-label">{sub}</div>',
                       unsafe_allow_html=True)
            info_btn(ik, b)
    for w in r["warnings"]:
        st.warning(w)
    with st.expander("See the first few readings"):
        prev = df.head(8).rename(columns={"timestamp": "Time", "energy_kwh": "Units", "co2": "CO₂ (t)",
                                          "power_factor": "Efficiency score", "load_type": "How busy"})
        st.markdown(prev.to_html(index=False, classes="tbl", border=0), unsafe_allow_html=True)
    ask_box("data", "the data check",
            {"report": {k: r[k] for k in ("rows", "interval_minutes", "coverage_pct", "missing_intervals",
                                          "duplicates_removed", "missing_energy", "warnings")},
             "columns_used": r["columns_used"]},
            ["Is my data reliable?", "What if readings are missing?", "Where does the data come from?"])

# ================================================================== EVALUATION
elif page == PAGES[7]:
    header("Evaluation", "How we tested ShiftWatt: accuracy, AI mistakes, alerts, translation and speed")

    @st.cache_data(show_spinner=False)
    def load_eval(stamp: float):
        from evaluate import OUT, main as run_eval
        if not OUT.exists():
            run_eval()
        return json.loads(OUT.read_text(encoding="utf-8"))

    from evaluate import OUT as EVAL_OUT
    ev = load_eval(EVAL_OUT.stat().st_mtime if EVAL_OUT.exists() else 0)
    acc, hd, ab, tr, lat = (ev["accuracy"], ev["hallucination_detection"], ev["alert_benchmark"],
                            ev["translation_safety"], ev["latency_ms"])
    _v2 = hd["after_v2_shift_scoped_check"]
    read_aloud(f"Our calculations match an independent recount with zero error. When we deliberately damaged AI "
               f"summaries, our checker caught {_v2['number_off_by_5pct_detected_pct']:.0f} percent of wrong numbers and "
               f"{_v2['number_swapped_between_shifts_detected_pct']:.0f} percent of numbers given to the wrong shift, with "
               f"no false alarms. Unusual usage in quiet hours was found every time. All 35,040 readings are processed "
               f"in {lat['total_ms'] / 1000:.1f} seconds.", "eval")
    a1, a2, a3, a4 = st.columns(4)
    for col, (ttl, val, sub) in zip((a1, a2, a3, a4), [
        ("Calculation error", f"{acc['max_abs_error_kwh']:g} units", "vs an independent recount"),
        ("AI mistakes caught", f"{hd['after_v2_shift_scoped_check']['number_off_by_5pct_detected_pct']:.0f}%",
         "numbers off by 5% (300 tests)"),
        ("Correct text wrongly flagged", f"{hd['after_v2_shift_scoped_check']['clean_outputs_wrongly_flagged_pct']:g}%",
         "no false alarms in 300 tests"),
        ("Time for 35,040 readings", f"{lat['total_ms'] / 1000:.2f} s", "load, analyse, alerts, summary, check"),
    ]):
        with col, st.container(border=True):
            st.markdown(f'<div class="mini-label">{ttl}</div><div class="big">{val}</div><div class="mini-label">{sub}</div>',
                        unsafe_allow_html=True)

    with st.container(border=True):
        card_title("Does the checker catch AI mistakes?", "Share of deliberately damaged summaries that were caught. "
                   "v2 checks each number against the shift it is written next to.", "evalinfo")
        lab = {"number_off_by_2pct_detected_pct": "Number off by 2%", "number_off_by_5pct_detected_pct": "Number off by 5%",
               "number_off_by_20pct_detected_pct": "Number off by 20%", "number_off_by_50pct_detected_pct": "Number off by 50%",
               "number_swapped_between_shifts_detected_pct": "Number given to wrong shift",
               "invented_cause_or_prediction_detected_pct": "Invented cause / prediction"}
        rows = [{"Mistake": lab[k], "Checker": name, "Caught %": hd[key][k]} for k in lab
                for key, name in (("before_v1_check_against_all_values", "v1 (before)"),
                                  ("after_v2_shift_scoped_check", "v2 (now)"))]
        f = px.bar(pd.DataFrame(rows), x="Mistake", y="Caught %", color="Checker", barmode="group", text_auto=".0f",
                   color_discrete_map={"v1 (before)": T["muted"], "v2 (now)": "#22C55E"}, labels={"Mistake": ""})
        f.update_traces(textposition="outside", cliponaxis=False)
        f.update_yaxes(range=[0, 112])
        plot(f, 330)

    with st.container(border=True):
        card_title("Does the alert system catch problems?",
                   f"{ab['injected_anomalies']} fake problems added to the real data. Quiet hours = times the plant is "
                   "normally quiet (like nights and weekends); busy hours = normal working times.", "evalinfo")
        mrows = []
        for name, m in ab["methods"].items():
            short = name.replace("ShiftWatt – ", "ShiftWatt: ").replace("Baseline – ", "Simple: ")
            mrows += [{"Method": short, "Where": "Quiet hours", "Found %": m["detected_in_quiet_hours_pct"]},
                      {"Method": short, "Where": "Busy hours", "Found %": m["detected_in_busy_hours_pct"]}]
        f = px.bar(pd.DataFrame(mrows), y="Method", x="Found %", color="Where", barmode="group", orientation="h",
                   text_auto=".0f", color_discrete_map={"Quiet hours": "#1BA3B8", "Busy hours": "#F5C242"},
                   labels={"Method": ""})
        f.update_xaxes(range=[0, 110])
        f.update_yaxes(autorange="reversed")
        plot(f, 360)
        st.markdown('<div class="note">▸ ShiftWatt finds <b>every</b> fake problem in quiet hours (e.g. machines left on at '
                    'night), where a single fixed limit finds <b>none</b>.<br>▸ In busy hours, normal usage already varies a '
                    'lot from day to day, so medium-sized jumps look normal; only large changes are flagged. A production '
                    'schedule would improve this.</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1, st.container(border=True):
        card_title("Translation safety", "A number changed during translation is caught")
        st.markdown("".join(f'<div class="point">{"Hindi" if k == "hi" else "Kannada"}: correct translation passes '
                            f'{"✓" if v["correct_translation_passes"] else "✗"} · changed numbers caught '
                            f'<b>{v["changed_number_detected_pct"]:.0f}%</b></div>' for k, v in tr.items()),
                    unsafe_allow_html=True)
    with c2, st.container(border=True):
        card_title("Speed", "Milliseconds for 35,040 readings")
        names = {"load_and_clean": "Load + clean", "assign_shifts": "Assign shifts", "shift_stats": "Shift numbers",
                 "alert_scoring": "Unusual usage", "write_brief_builtin": "Write summary", "verify_brief": "Check summary"}
        f = px.bar(x=[lat[k] for k in names], y=list(names.values()), orientation="h", text_auto=".0f",
                   labels={"x": "ms", "y": ""}, color_discrete_sequence=["#1BA3B8"])
        f.update_yaxes(autorange="reversed")
        plot(f, 230, legend=False)

    with st.container(border=True):
        card_title("Compare AI models and prompts (live)",
                   "Runs the AI 3 times with the basic prompt (v1) and our optimised prompt (v2) and measures valid "
                   "output, wrong numbers, invented causes, technical jargon and speed. Needs Gemini or Ollama.")
        if st.button("▶ Run the AI comparison", type="primary"):
            from evaluate import run_llm_benchmark
            with st.spinner("Asking the AI 6 times (about 30 seconds, with short pauses for the free limit)…"):
                try:
                    st.session_state["llm_bench"] = run_llm_benchmark(facts, cfg["provider"], runs=3)
                except Exception as e:  # noqa: BLE001
                    st.session_state["llm_bench"] = str(e)
        lb = st.session_state.get("llm_bench")
        if isinstance(lb, str):
            st.warning(lb)
        elif lb:
            tb = pd.DataFrame(lb).rename(columns={
                "model": "AI", "prompt": "Prompt", "runs": "Runs", "first_try_ok_pct": "OK first try %",
                "valid_json_pct": "OK after 1 retry %", "problems_seen": "Problems seen",
                "fully_verified_pct": "All numbers correct %", "wrong_numbers_per_run": "Wrong numbers / run",
                "cause_or_prediction_words_per_run": "Invented causes / run", "technical_jargon_per_run": "Jargon words / run",
                "avg_latency_s": "Seconds / run", "observations_per_run": "Points / run"})
            st.markdown(tb.to_html(index=False, classes="tbl", border=0), unsafe_allow_html=True)
            st.download_button("Download results (.csv)", tb.to_csv(index=False), "ai_comparison.csv")

    with st.container(border=True):
        card_title("Responsible AI safeguards", "What stops the system from misleading people")
        st.markdown("".join(f'<div class="point">✓ {x}</div>' for x in [
            "<b>No made-up numbers:</b> Python calculates everything; the AI only explains; every number is checked "
            "against the shift it is written next to.",
            "<b>No guessing causes or predicting:</b> the prompt forbids it and the checker flags words like "
            "'because' or 'will'.",
            "<b>No blame:</b> results never point at workers or specific machines.",
            "<b>Clear assumptions:</b> shift times, the price per unit and alert limits are always shown as the user's settings.",
            "<b>Honest data:</b> missing readings are reported, never filled in.",
            "<b>Privacy:</b> can run fully offline (Ollama or built-in writer); only summary numbers, never raw meter "
            "data, are sent to an online AI.",
            "<b>People decide:</b> alerts and summaries inform; they never switch machines on or off.",
            "<b>Free and transparent:</b> every chart has a 'How is this calculated?' button."]), unsafe_allow_html=True)

    if st.button("↻ Re-run all tests (about 5 seconds)"):
        from evaluate import main as run_eval
        with st.spinner("Running tests…"):
            run_eval()
        st.rerun()
    ask_box("eval", "the evaluation",
            {"accuracy": acc["max_abs_error_kwh"], "checker": hd, "alerts": ab, "translation": tr, "speed_ms": lat,
             "ai_comparison": lb if isinstance(lb, list) else "not run yet"},
            ["What do these tests show?", "Why are busy-hour alerts lower?", "What is v1 vs v2?"])

# ================================================================== SETTINGS
elif page == PAGES[8]:
    header("Settings", "Change these and every page updates")
    read_aloud("Here you can choose your data, set your shift times, set the price per unit, and add a free AI helper "
               "and phone alerts. Every page updates as soon as you change something.", "settings")
    with st.container(border=True):
        card_title("1. Your data", "Use the sample steel-factory data, or upload your own meter file (CSV)")
        src = st.radio("Data", ["Sample data", "My own file"], index=0 if cfg["upload"] is None else 1,
                       horizontal=True, label_visibility="collapsed")
        if src == "Sample data" and cfg["upload"] is not None:
            cfg["upload"], cfg["upload_name"] = None, None
            st.rerun()
        if src == "My own file":
            up = st.file_uploader("Your file needs a time column and an electricity column", type="csv")
            if up is not None and up.name != cfg["upload_name"]:
                cfg["upload"], cfg["upload_name"] = up.getvalue(), up.name
                st.rerun()
            if cfg["upload_name"]:
                st.caption(f"Using: {cfg['upload_name']}")

    with st.container(border=True):
        card_title("2. Your shift times", "When does each shift start and end? A night shift can end after midnight "
                   "(e.g. 10 PM to 6 AM). Changes apply to every page straight away.")
        n = st.number_input("Number of shifts", 1, 4, len(cfg["shifts"]),
                            help="Changing this splits the day evenly; you can then adjust the times.")
        if n != len(cfg["shifts"]):
            split_evenly(int(n), cfg["shifts"][0]["start"])
            st.rerun()
        for i, s in enumerate(cfg["shifts"]):
            a, b, c = st.columns([2, 1, 1])
            s["name"] = a.text_input("Name", s["name"], key=f"sn{i}", label_visibility="collapsed" if i else "visible")
            s["start"] = b.time_input("Starts", s["start"], key=f"ss{i}", step=1800,
                                      label_visibility="collapsed" if i else "visible")
            s["end"] = c.time_input("Ends", s["end"], key=f"se{i}", step=1800,
                                    label_visibility="collapsed" if i else "visible")
        problems = validate_shifts([{"name": x["name"], "start": x["start"].strftime("%H:%M"),
                                     "end": x["end"].strftime("%H:%M")} for x in cfg["shifts"]])
        for msg in problems:
            st.warning("⚠️ " + msg.replace("Shifts overlap", "Some shifts overlap").replace("Unassigned", "not counted")
                       + " Overlapping time is counted for the shift listed first.")
        prev = go.Figure()
        for i, x in enumerate(cfg["shifts"]):
            a0, b0 = x["start"].hour + x["start"].minute / 60, x["end"].hour + x["end"].minute / 60
            spans = [(a0, b0)] if a0 < b0 else ([(a0, 24), (0, b0)] if b0 > 0 else [(a0, 24)])
            for x0, x1 in spans:
                prev.add_trace(go.Bar(y=[x["name"]], x=[x1 - x0], base=[x0], orientation="h", showlegend=False,
                                      marker_color=SHIFT_COLORS[i % 4],
                                      hovertemplate=f"{x['name']}: {t12(x['start'])} – {t12(x['end'])}<extra></extra>"))
        prev.update_xaxes(range=[0, 24], tickvals=list(range(0, 25, 2)),
                          ticktext=[t12(dt.time(h % 24)).replace(" ", "") for h in range(0, 25, 2)])
        prev.update_yaxes(autorange="reversed")
        prev.update_layout(barmode="overlay", bargap=0.25)
        plot(prev, 70 + 34 * len(cfg["shifts"]), legend=False)
        b1, b2 = st.columns(2)
        if b1.button("Split the day evenly", width="stretch"):
            split_evenly(len(cfg["shifts"]), cfg["shifts"][0]["start"])
            st.rerun()
        if b2.button("Reset to 3 shifts of 8 hours", width="stretch"):
            del st.session_state["cfg"]
            clear_shift_widgets()
            st.rerun()

    with st.container(border=True):
        card_title("3. Electricity price", "Used only to estimate your bill")
        cfg["price"] = st.number_input("₹ per unit", 0.0, 50.0, float(cfg["price"]), 0.5)

    with st.container(border=True):
        card_title("4. AI helper (optional, free)",
                   "Without this, a built-in writer is used. With it, answers and summaries are richer.")
        opts = {"auto": "Automatic (recommended)", "gemini": "Google Gemini (free, needs internet)",
                "ollama": "Ollama (free, runs on this computer)", "template": "Built-in only (no AI)"}
        cfg["provider"] = st.selectbox("AI", list(opts), index=list(opts).index(cfg["provider"]),
                                       format_func=opts.get, label_visibility="collapsed")
        key = st.text_input("Gemini key (free from aistudio.google.com)", type="password",
                            placeholder="paste key here" if not os.environ.get("GEMINI_API_KEY") else "key saved ✓")
        if key:
            os.environ["GEMINI_API_KEY"] = key.strip()
        if st.button("🔍 Find my Gemini models", help="Asks Google which models your key can use"):
            try:
                from google import genai as _g
                _client = _g.Client(api_key=os.environ["GEMINI_API_KEY"])  # keep a reference while listing
                ms = [m.name.replace("models/", "") for m in list(_client.models.list())
                      if "generateContent" in (getattr(m, "supported_actions", None) or ["generateContent"])
                      and "gemini" in m.name]
                st.session_state["gemini_models"] = sorted(ms, key=lambda n: (("flash-lite" not in n), ("flash" not in n), n))
                os.environ["GEMINI_FALLBACKS"] = ",".join(
                    [m for m in st.session_state["gemini_models"] if "flash" in m and "preview" not in m
                     and "image" not in m and "tts" not in m and "live" not in m][:4])
            except KeyError:
                st.error("Paste your Gemini key above first.")
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not list models: {type(e).__name__}: {str(e)[:200]}")
        found = st.session_state.get("gemini_models")
        m1, m2 = st.columns([3, 1], vertical_alignment="bottom")
        if found:
            cur = os.environ.get("GEMINI_MODEL", found[0])
            gm = m1.selectbox("Gemini model (Flash-Lite models are listed first: fast and free)", found,
                              index=found.index(cur) if cur in found else 0)
        else:
            gm = m1.text_input("Gemini model name (or click 'Find my Gemini models')",
                               os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite"))
        if gm.strip():
            os.environ["GEMINI_MODEL"] = gm.strip()
        if m2.button("Test connection", width="stretch"):
            from llm_notes import PROVIDERS as _P, _parse_json_text as _pj
            prov = resolve_provider(cfg["provider"])
            if prov not in _P:
                st.error("No AI found: paste a Gemini key above (or start Ollama).")
            else:
                import time as _t
                t0 = _t.perf_counter()
                try:
                    out = _pj(_P[prov]('Reply only with JSON {"ok": true, "model": "<your name>"}', "Say hello."))
                    st.success(f"✅ {prov.title()} is working ({_t.perf_counter() - t0:.1f} s). Reply: {out}")
                except Exception as e:  # noqa: BLE001
                    st.error(f"❌ {prov.title()} failed: {type(e).__name__}: {str(e)[:300]}")
        now = resolve_provider(cfg["provider"])
        st.caption(("🟢 Ollama found: " + (ollama_model() or "") if ollama_models() else "⚪ Ollama not found on this computer")
                   + " · Now using: " + {"gemini": "Gemini", "ollama": "Ollama"}.get(now, "built-in writer"))

    with st.container(border=True):
        card_title("5. Phone alerts (optional, free)",
                   "Get live-monitor alerts on your phone through Telegram. In Telegram: message @BotFather, send "
                   "/newbot and copy the bot token; then open your new bot and press Start; message @userinfobot to get "
                   "your chat ID.")
        t1, t2 = st.columns(2)
        cfg["tg_token"] = t1.text_input("Bot token", cfg.get("tg_token", ""), type="password")
        cfg["tg_chat"] = t2.text_input("Your chat ID", cfg.get("tg_chat", ""))
        if st.button("Send a test message"):
            err = send_telegram(cfg["tg_token"], cfg["tg_chat"], "✅ ShiftWatt test: phone alerts are working.", wait=True)
            (st.error(f"Could not send: {err}") if err else st.success("Sent! Check your Telegram."))

    with st.expander("Advanced"):
        cfg["label_end"] = st.checkbox("Each meter time marks the END of its 15 minutes (true for the sample data)",
                                       cfg["label_end"])
    if cfg_shifts() != shifts:  # apply changes to every page immediately
        st.rerun()

st.markdown('<p class="note" style="text-align:center;margin-top:24px">ShiftWatt shows what the meter recorded. '
            "It does not guess causes or predict the future. Shift times are set by you.</p>", unsafe_allow_html=True)
