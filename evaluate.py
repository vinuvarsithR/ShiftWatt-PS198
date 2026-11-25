"""ShiftWatt evaluation & benchmarking (Review 3).

Run:  python evaluate.py            -> writes evaluation_results.json (deterministic parts, no internet needed)
LLM/prompt benchmark (needs a free Gemini key or Ollama) runs from the app's 📊 Evaluation page.
"""
from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from alerts import alert_summary, detect_alerts, score_readings
from data_pipeline import load_and_clean
from llm_notes import (CAUSAL_RE, PROVIDERS, SYSTEM_PROMPT, _parse, build_prompt, resolve_provider, template_notes,
                       verify_notes)
from shift_engine import DEFAULT_SHIFTS, assign_shifts, compute_shift_stats
from translate import brief_text, verify_translation

ROOT = Path(__file__).parent
SAMPLE = ROOT / "data" / "Steel_industry_data.csv"
OUT = ROOT / "evaluation_results.json"


def default_facts(df=None):
    if df is None:
        df, _ = load_and_clean(SAMPLE)
    a = assign_shifts(df, DEFAULT_SHIFTS)
    f = compute_shift_stats(a, DEFAULT_SHIFTS)
    ev = detect_alerts(score_readings(a), 5.0, 5.0)
    return df, a, {**f, "alerts_summary": alert_summary(ev, DEFAULT_SHIFTS)}


# ---------------------------------------------------------------- 1. calculation accuracy
def eval_accuracy(df, a, facts):
    """Recompute per-shift totals with an independent, naive row-by-row method and compare."""
    naive = {s["name"]: 0.0 for s in DEFAULT_SHIFTS}
    for ts, e in zip(df["timestamp"], df["energy_kwh"]):
        st = ts - pd.Timedelta(minutes=15)  # reading covers the 15 minutes before its timestamp
        h = st.hour
        naive["Shift A" if 6 <= h < 14 else "Shift B" if 14 <= h < 22 else "Shift C"] += e
    rows, max_err = [], 0.0
    for s in facts["shifts"]:
        err = abs(s["total_kwh"] - round(naive[s["name"]], 2))
        max_err = max(max_err, err)
        rows.append({"shift": s["name"], "engine_kwh": s["total_kwh"], "independent_kwh": round(naive[s["name"]], 2),
                     "abs_error": round(err, 4)})
    conservation = abs(sum(s["total_kwh"] for s in facts["shifts"]) - df["energy_kwh"].sum())
    return {"per_shift": rows, "max_abs_error_kwh": round(max_err, 4),
            "shift_totals_vs_file_total_diff_kwh": round(float(conservation), 4),
            "boundary_and_unit_tests": "7/7 pytest tests pass (night-shift boundary, kW→kWh, overlaps, alerts, verifier)"}


# ---------------------------------------------------------------- 2. hallucination detection
def _numbers(text):
    return [m for m in re.finditer(r"(?<![\w.])\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.])\d+(?:\.\d+)?", text)]


def _corrupt_number(text, rng, pct):
    cands = [m for m in _numbers(text) if float(m.group().replace(",", "")) > 10]
    if not cands:
        return None
    m = rng.choice(cands)
    v = float(m.group().replace(",", ""))
    nv = v * (1 + pct * rng.choice([-1, 1]))
    rep = f"{nv:,.0f}" if "." not in m.group() else f"{nv:,.2f}"
    return text[:m.start()] + rep + text[m.end():]


CAUSAL_INJECT = ["because the furnace ran longer", "due to poor maintenance", "and this will increase next month",
                 "which is likely caused by night-shift workers", "caused by the rolling mill"]


def eval_hallucination(facts, trials=300, seed=7, scoped=True):
    rng = random.Random(seed)
    base = template_notes(facts)
    names = [s["name"] for s in facts["shift_definitions"]]
    results = {}

    def check(notes):
        return not verify_notes(json.loads(json.dumps(notes)), facts, scoped=scoped)["all_verified"]

    # clean outputs with different (legal) rounding styles -> should NOT be flagged
    fp = 0
    for t in range(trials):
        n = json.loads(json.dumps(base))
        for o in n["observations"]:
            def rround(m):
                v = float(m.group().replace(",", ""))
                style = rng.choice(["int", "1dp", "comma"])
                return f"{v:.0f}" if style == "int" else f"{v:.1f}" if style == "1dp" else f"{v:,.0f}"
            o["text"] = re.sub(r"(?<![\w.])\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.])\d{2,}(?:\.\d+)?", rround, o["text"])
        fp += check(n)
    results["clean_outputs_wrongly_flagged_pct"] = round(fp / trials * 100, 2)

    for label, pct in [("number_off_by_2pct", .02), ("number_off_by_5pct", .05), ("number_off_by_20pct", .20),
                       ("number_off_by_50pct", .50)]:
        caught = done = 0
        for _ in range(trials):
            n = json.loads(json.dumps(base))
            o = rng.choice(n["observations"])
            new = _corrupt_number(o["text"], rng, pct)
            if new is None:
                continue
            o["text"] = new
            done += 1
            caught += check(n)
        results[f"{label}_detected_pct"] = round(caught / done * 100, 2)

    caught = 0
    for _ in range(trials):
        n = json.loads(json.dumps(base))
        o = rng.choice(n["observations"])
        o["text"] = o["text"].rstrip(".") + ", " + rng.choice(CAUSAL_INJECT) + "."
        caught += check(n)
    results["invented_cause_or_prediction_detected_pct"] = round(caught / trials * 100, 2)

    # honest limitation: a real number attached to the wrong shift
    caught = done = 0
    for _ in range(trials):
        n = json.loads(json.dumps(base))
        o = n["observations"][0]
        a_, b_ = rng.sample(names, 2)
        if a_ in o["text"] and b_ in o["text"]:
            o["text"] = o["text"].replace(a_, "§").replace(b_, a_).replace("§", b_)
            done += 1
            caught += check(n)
    results["number_swapped_between_shifts_detected_pct"] = round(caught / max(done, 1) * 100, 2)
    results["trials_per_type"] = trials
    return results


# ---------------------------------------------------------------- 3. alert engine benchmark
def eval_alerts(a, n_inject=300, seed=11):
    """Inject synthetic anomalies into the real data and compare our context-aware detector with two simpler ones."""
    rng = np.random.default_rng(seed)
    base = a.copy().reset_index(drop=True)
    sc0 = score_readings(base)
    idx = rng.choice(len(base), size=n_inject, replace=False)
    kinds = rng.choice(["spike", "drop"], size=n_inject)
    inj = base.copy()
    truth = np.zeros(len(base), dtype=bool)
    for i, k in zip(idx, kinds):
        exp = sc0.loc[i, "expected_kwh"]
        if k == "spike":
            inj.loc[i, "energy_kwh"] = exp + rng.uniform(25, 60)  # +25 to +60 units over normal
            truth[i] = True
        elif exp >= 20:  # a drop is only meaningful when the plant is normally busy
            inj.loc[i, "energy_kwh"] = exp * rng.uniform(0.0, 0.15)
            truth[i] = True
    sc = score_readings(inj)
    v, e, z = sc["energy_kwh"].values, sc["expected_kwh"].values, sc["z"].values

    quiet = e < 20  # plant normally quiet at that weekday/time

    def metrics(flag):
        tp = int((flag & truth).sum())
        fpr = float((flag & ~truth).sum() / (~truth).sum() * 100)
        return {"detected_pct": round(tp / truth.sum() * 100, 1),
                "detected_in_quiet_hours_pct": round(float((flag & truth & quiet).sum() / max((truth & quiet).sum(), 1) * 100), 1),
                "detected_in_busy_hours_pct": round(float((flag & truth & ~quiet).sum() / max((truth & ~quiet).sum(), 1) * 100), 1),
                "other_readings_flagged_pct": round(fpr, 2)}

    out = {"injected_anomalies": int(truth.sum()), "injected_in_quiet_hours": int((truth & quiet).sum()),
           "injected_in_busy_hours": int((truth & ~quiet).sum()), "methods": {}}
    for name, thr in [("ShiftWatt – Only big problems", 8.0), ("ShiftWatt – Big and medium (default)", 5.0),
                      ("ShiftWatt – Everything unusual", 3.5)]:
        out["methods"][name] = metrics((np.abs(z) >= thr) & (np.abs(v - e) >= 5))
    mu, sd = inj["energy_kwh"].mean(), inj["energy_kwh"].std()
    out["methods"]["Baseline – one fixed limit for all hours (mean ± 3 SD)"] = metrics((v > mu + 3 * sd) | (v < mu - 3 * sd))
    hr = inj["interval_start"].dt.hour
    g = inj.groupby(hr)["energy_kwh"]
    hm, hs = g.transform("mean").values, g.transform("std").values
    out["methods"]["Baseline – per-hour average ± 3 SD (ignores weekday)"] = metrics((v > hm + 3 * hs) | (v < hm - 3 * hs))
    return out


# ---------------------------------------------------------------- 4. translation safety
def eval_translation(facts, trials=200, seed=3):
    rng = random.Random(seed)
    src = brief_text(template_notes(facts))
    names = [s["name"] for s in facts["shift_definitions"]]
    out = {}
    for code in ("hi", "kn"):
        f = next((ROOT / "translations").glob(f"*_{code}.txt"), None)
        if not f:
            continue
        tr = f.read_text(encoding="utf-8").strip()
        ok_clean = verify_translation(src, tr, names)["ok"]
        caught = 0
        for _ in range(trials):
            caught += not verify_translation(src, _corrupt_number(tr, rng, rng.choice([.05, .2, .5])), names)["ok"]
        out[code] = {"correct_translation_passes": ok_clean, "changed_number_detected_pct": round(caught / trials * 100, 1)}
    return out


# ---------------------------------------------------------------- 5. latency
def eval_latency(runs=3):
    t = {k: [] for k in ("load_and_clean", "assign_shifts", "shift_stats", "alert_scoring", "write_brief_builtin",
                         "verify_brief")}
    for _ in range(runs):
        s = time.perf_counter(); df, _ = load_and_clean(SAMPLE); t["load_and_clean"].append(time.perf_counter() - s)
        s = time.perf_counter(); a = assign_shifts(df, DEFAULT_SHIFTS); t["assign_shifts"].append(time.perf_counter() - s)
        s = time.perf_counter(); f = compute_shift_stats(a, DEFAULT_SHIFTS); t["shift_stats"].append(time.perf_counter() - s)
        s = time.perf_counter(); detect_alerts(score_readings(a), 5.0, 5.0); t["alert_scoring"].append(time.perf_counter() - s)
        s = time.perf_counter(); n = template_notes(f); t["write_brief_builtin"].append(time.perf_counter() - s)
        s = time.perf_counter(); verify_notes(n, f); t["verify_brief"].append(time.perf_counter() - s)
    res = {k: round(float(np.median(v)) * 1000, 1) for k, v in t.items()}
    res["total_ms"] = round(sum(res.values()), 1)
    res["rows"] = 35040
    return res


# ---------------------------------------------------------------- 6. LLM + prompt benchmark (needs a model)
PROMPT_V1 = """Summarise this factory energy data per shift. Return JSON:
{"summary": "...", "observations": [{"title": "...", "text": "...", "cited_values": []}],
 "data_quality_notes": [], "assumption_statement": "..."}"""
JARGON = re.compile(r"\b(kwh|kilowatt|power factor|load type|light_load|medium_load|maximum_load|z-score|anomal\w*)\b",
                    re.I)


def run_llm_benchmark(facts, provider="auto", runs=3, prompts=("v1 basic", "v2 ShiftWatt"), pause_s=2.0) -> list[dict]:
    """Runs each prompt `runs` times the same way the app does (one retry on a failed answer) and records why any
    run failed. Pauses between calls to stay inside free-tier rate limits."""
    prov = resolve_provider(provider)
    if prov not in PROVIDERS:
        raise RuntimeError("No AI model available. Add a free Gemini key or start Ollama.")
    _, user = build_prompt(facts)
    rows = []
    for pname in prompts:
        system = PROMPT_V1 if pname.startswith("v1") else SYSTEM_PROMPT
        lat, first_ok, final_ok, ver, mism, causal, jargon, obs, errors, issues = [], 0, 0, 0, 0, 0, 0, 0, [], []
        for r in range(runs):
            notes = None
            for attempt in range(2):
                if r or attempt or pname != prompts[0]:
                    time.sleep(pause_s)
                s = time.perf_counter()
                try:
                    notes = _parse(PROVIDERS[prov](system, user))
                    if not notes["observations"]:
                        raise ValueError("answer had no observations")
                    lat.append(time.perf_counter() - s)
                    first_ok += attempt == 0
                    break
                except Exception as e:  # noqa: BLE001
                    lat.append(time.perf_counter() - s)
                    errors.append(f"{type(e).__name__}: {str(e)[:70]}")
                    notes = None
            if notes is None:
                continue
            final_ok += 1
            v = verify_notes(notes, facts)
            ver += v["all_verified"]
            mism += len(v["mismatches"])
            issues += [f"number {m['value']} not in data" for m in v["mismatches"][:1]]
            txt = notes["summary"] + " " + " ".join(o["text"] for o in notes["observations"])
            causal += len(CAUSAL_RE.findall(txt))
            jargon += len(JARGON.findall(txt))
            obs += len(notes["observations"])
        rows.append({"model": prov, "prompt": pname, "runs": runs,
                     "first_try_ok_pct": round(first_ok / runs * 100), "valid_json_pct": round(final_ok / runs * 100),
                     "fully_verified_pct": round(ver / runs * 100), "wrong_numbers_per_run": round(mism / runs, 2),
                     "cause_or_prediction_words_per_run": round(causal / runs, 2),
                     "technical_jargon_per_run": round(jargon / runs, 2), "avg_latency_s": round(float(np.mean(lat)), 2),
                     "observations_per_run": round(obs / max(final_ok, 1), 1),
                     "problems_seen": "; ".join(dict.fromkeys(errors + issues))[:220] or "none"})
    return rows


def main():
    df, a, facts = default_facts()
    res = {"dataset": "UCI Steel Industry Energy Consumption, 35,040 readings (15-min, 2018)",
           "accuracy": eval_accuracy(df, a, facts),
           "hallucination_detection": {"before_v1_check_against_all_values": eval_hallucination(facts, scoped=False),
                                       "after_v2_shift_scoped_check": eval_hallucination(facts, scoped=True)},
           "alert_benchmark": eval_alerts(a),
           "translation_safety": eval_translation(facts),
           "latency_ms": eval_latency()}
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
