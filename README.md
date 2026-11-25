# ⚡ ShiftWatt – Manufacturing Shift Trend Brief (PS 198)

A simple dashboard that shows how much electricity each work shift uses, spots unusual usage, alerts you live,
and writes a plain-language summary you can trust. **Python calculates every number, the AI only explains them,
and a checker verifies every number the AI writes.** Everything used is free.

## Run on your computer
Windows: double-click `run_shiftwatt.bat`. Or:
```bash
pip install -r requirements.txt
streamlit run app.py          # opens http://localhost:8501
```
AI (optional, free): paste a Gemini key from aistudio.google.com in ⚙️ Settings, or install Ollama and run
`ollama pull llama3.2`. Without either, a built-in writer is used, so the app always works offline.

## Deploy
| Option | Steps |
|---|---|
| Docker | `docker build -t shiftwatt .` then `docker run -p 8501:8501 -e GEMINI_API_KEY=... shiftwatt` |
| Docker + offline AI | `docker compose up -d`, then once: `docker compose exec ollama ollama pull llama3.2` |
| Streamlit Community Cloud (free) | Push to GitHub → share.streamlit.io → New app → `app.py`; add `GEMINI_API_KEY` under Secrets |

Keys are read from environment variables or Streamlit secrets – never stored in code. A health check is built into
the Docker image, and GitHub Actions runs the tests and evaluation on every push.

## Pages
Home · Unusual usage · 📡 Live monitor (simulation with pop-up, desktop and Telegram alerts) · Daily pattern ·
Pollution & efficiency · Written summary (9 languages, read aloud) · Data check · 📊 Evaluation · Settings.
Every chart has a ⓘ "How is this calculated?" button and every page has a "Have a question?" helper.

## Code
| File | What it does |
|---|---|
| `data_pipeline.py` | Load and clean the meter CSV, detect units, find gaps |
| `shift_engine.py` | Assign readings to shifts (incl. night shifts) and compute per-shift numbers |
| `alerts.py` | Unusual usage vs the same weekday and time (median + MAD) |
| `llm_notes.py` | Grounded prompt, Gemini / Ollama / built-in writer, shift-scoped number checker |
| `assistant.py` | "Have a question?" helper |
| `translate.py`, `voice.py` | 8 Indian languages with number re-check; read aloud |
| `notify.py` | Desktop notifications, beep, Telegram phone alerts |
| `evaluate.py` | Accuracy, AI-mistake detection, alert benchmark, translation safety, speed, AI/prompt comparison |
| `app.py` | Streamlit dashboard |

Tests: `python -m pytest -q tests` (8 tests). Evaluation: `python evaluate.py` → `evaluation_results.json`.

## Data
UCI Steel Industry Energy Consumption (2018, 35,040 readings every 15 minutes). Any CSV with a time column and an
energy column also works.
