"""Smart alert system: context-aware detection of abnormally HIGH or LOW electricity use.

Each reading is compared with what is TYPICAL for that exact weekday and time slot
(e.g. "Thursday 17:15"), using a robust score (median + MAD), so a busy Monday morning is not
flagged just for being busy and a quiet Sunday night is not flagged for being quiet.
Consecutive abnormal readings are grouped into one alert event. No causes are inferred.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def score_readings(df: pd.DataFrame) -> pd.DataFrame:
    """Add expected (typical) kWh, robust z-score and deviation % for every reading.

    df needs: timestamp, interval_start, energy_kwh, shift (output of shift_engine.assign_shifts).
    """
    out = df.copy()
    t = out["interval_start"]
    out["dow"] = t.dt.dayofweek
    out["slot"] = t.dt.hour * 60 + t.dt.minute
    g = out.groupby(["dow", "slot"])["energy_kwh"]
    out["expected_kwh"] = g.transform("median")
    mad = g.transform(lambda s: (s - s.median()).abs().median())
    # MAD can be 0 for very flat slots; use a floor of 5% of typical (min 0.5 kWh) so one outlier
    # cannot inflate its own yardstick (a plain std fallback would hide the spike)
    scale = np.maximum(1.4826 * mad, np.maximum(0.05 * out["expected_kwh"].abs(), 0.5))
    out["z"] = (out["energy_kwh"] - out["expected_kwh"]) / scale
    out["deviation_pct"] = np.where(
        out["expected_kwh"] > 0, (out["energy_kwh"] - out["expected_kwh"]) / out["expected_kwh"] * 100, np.nan
    )
    return out


def detect_alerts(scored: pd.DataFrame, z_threshold: float = 3.5, min_abs_kwh: float = 5.0,
                  interval_minutes: int = 15) -> pd.DataFrame:
    """Group abnormal readings into alert events.

    A reading is abnormal when |z| >= z_threshold AND it differs from typical by >= min_abs_kwh
    (avoids alerts on tiny night-time wobbles). Severity: critical when |z| >= 2 x threshold.
    """
    s = scored.dropna(subset=["energy_kwh"]).sort_values("timestamp")
    diff = (s["energy_kwh"] - s["expected_kwh"]).abs()
    flag = (s["z"].abs() >= z_threshold) & (diff >= min_abs_kwh)
    a = s[flag].copy()
    if a.empty:
        return pd.DataFrame(columns=["severity", "direction", "shift", "start", "end", "duration_min", "readings",
                                     "worst_kwh", "typical_kwh", "deviation_pct", "max_z", "message"])
    a["direction"] = np.where(a["z"] > 0, "HIGH", "LOW")
    step = pd.Timedelta(minutes=interval_minutes)
    new_event = (a["timestamp"].diff() != step) | (a["direction"] != a["direction"].shift()) | (
        a["shift"] != a["shift"].shift())
    a["event"] = new_event.cumsum()

    rows = []
    for _, e in a.groupby("event"):
        worst = e.loc[e["z"].abs().idxmax()]
        start, end = e["interval_start"].min(), e["timestamp"].max()
        max_z = float(e["z"].abs().max())
        sev = "CRITICAL" if max_z >= 2 * z_threshold else "WARNING"
        dev = float(worst["deviation_pct"]) if pd.notna(worst["deviation_pct"]) else None
        wt = worst["interval_start"]
        dev_txt = f" ({dev:+.0f}%)" if dev is not None else ""
        span = f"{start:%d %b %H:%M}–{end:%H:%M}" if start.date() == end.date() else f"{start:%d %b %H:%M} – {end:%d %b %H:%M}"
        msg = (f"{sev} {worst['direction']} usage in {worst['shift']}, {span} {start:%Y}. "
               f"Most abnormal reading: {worst['energy_kwh']:.1f} kWh at {wt:%a %H:%M}, against a typical "
               f"{worst['expected_kwh']:.1f} kWh for that weekday and time{dev_txt}.")
        rows.append({
            "severity": sev, "direction": worst["direction"], "shift": worst["shift"],
            "start": start, "end": end, "duration_min": int((end - start).total_seconds() // 60),
            "readings": len(e), "worst_kwh": round(float(worst["energy_kwh"]), 2),
            "typical_kwh": round(float(worst["expected_kwh"]), 2),
            "deviation_pct": round(dev, 1) if dev is not None else None,
            "max_z": round(max_z, 1), "message": msg,
        })
    ev = pd.DataFrame(rows)
    ev["_rank"] = (ev["severity"] == "CRITICAL").astype(int)
    return ev.sort_values(["_rank", "max_z"], ascending=False).drop(columns="_rank").reset_index(drop=True)


def alert_summary(events: pd.DataFrame, shifts: list[dict]) -> dict:
    """Counts per shift for the facts JSON (so the LLM can mention them and the verifier can check them)."""
    out = {}
    for s in shifts:
        e = events[events["shift"] == s["name"]] if len(events) else events
        out[s["name"]] = {
            "high_alerts": int((e["direction"] == "HIGH").sum()) if len(e) else 0,
            "low_alerts": int((e["direction"] == "LOW").sum()) if len(e) else 0,
            "critical_alerts": int((e["severity"] == "CRITICAL").sum()) if len(e) else 0,
        }
    return out


def day_view(scored: pd.DataFrame, date) -> pd.DataFrame:
    """Readings of one day with the typical band, for the replay chart."""
    d = scored[scored["interval_start"].dt.date == pd.Timestamp(date).date()]
    return d[["timestamp", "interval_start", "energy_kwh", "expected_kwh", "z", "shift"]]
