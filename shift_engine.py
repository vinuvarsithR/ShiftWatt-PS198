"""Modules 3-4: shift assignment and per-shift analytics (owner: Monesh).

Pure computation: no LLM calls, no causal language.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ASSUMPTION_NOTE = "Shift timings are user-defined assumptions, not measured operating schedules."
DEFAULT_SHIFTS = [
    {"name": "Shift A", "start": "06:00", "end": "14:00"},
    {"name": "Shift B", "start": "14:00", "end": "22:00"},
    {"name": "Shift C", "start": "22:00", "end": "06:00"},
]


def _minutes(hhmm: str) -> int:
    h, m = str(hhmm).strip().split(":")[:2]
    h, m = int(h), int(m)
    if not (0 <= h <= 24 and 0 <= m < 60):
        raise ValueError(f"Invalid time '{hhmm}'")
    return (h * 60 + m) % 1440


def _duration_hours(s: dict) -> float:
    a, b = _minutes(s["start"]), _minutes(s["end"])
    return (((b - a) % 1440) or 1440) / 60


def _in_shift(mod: np.ndarray, s: dict) -> np.ndarray:
    a, b = _minutes(s["start"]), _minutes(s["end"])
    if a == b:
        return np.ones_like(mod, dtype=bool)
    if a < b:
        return (mod >= a) & (mod < b)
    return (mod >= a) | (mod < b)  # crosses midnight


def validate_shifts(shifts: list[dict]) -> list[str]:
    warns = []
    if not shifts:
        return ["No shifts defined."]
    names = [s["name"] for s in shifts]
    if len(set(names)) != len(names):
        warns.append("Shift names must be unique.")
    minutes = np.arange(1440)
    count = np.zeros(1440, dtype=int)
    for s in shifts:
        count += _in_shift(minutes, s)
    if (count > 1).any():
        warns.append(f"Shifts overlap for {int((count > 1).sum() / 60 * 10) / 10} hours of the day.")
    if (count == 0).any():
        warns.append(
            f"{int((count == 0).sum() / 60 * 10) / 10} hours of the day are not covered by any shift; "
            "those readings are marked Unassigned."
        )
    return warns


def assign_shifts(df: pd.DataFrame, shifts: list[dict], interval_minutes: int = 15,
                  label_at_end: bool = True) -> pd.DataFrame:
    """Add `interval_start` and `shift` columns.

    label_at_end=True: the timestamp marks the END of the interval (true for the steel dataset),
    so a 06:00 reading covers 05:45-06:00 and is assigned by its start time.
    First matching shift wins if shifts overlap.
    """
    out = df.copy()
    ref = out["timestamp"] - pd.Timedelta(minutes=interval_minutes) if label_at_end else out["timestamp"]
    out["interval_start"] = ref
    mod = (ref.dt.hour * 60 + ref.dt.minute).to_numpy()
    conds = [_in_shift(mod, s) for s in shifts]
    out["shift"] = np.select(conds, [s["name"] for s in shifts], default="Unassigned")
    return out


def _r(x, n=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return round(float(x), n)


def compute_shift_stats(df: pd.DataFrame, shifts: list[dict], interval_minutes: int = 15,
                        label_at_end: bool = True) -> dict:
    a = df if "shift" in df else assign_shifts(df, shifts, interval_minutes, label_at_end)
    start, end = a["interval_start"].min(), a["interval_start"].max()
    days = round((end - start + pd.Timedelta(minutes=interval_minutes)) / pd.Timedelta(days=1), 2)

    # Expected readings per shift from the full interval grid
    grid = pd.date_range(start, end, freq=f"{interval_minutes}min")
    gmod = grid.hour * 60 + grid.minute
    warnings = validate_shifts(shifts)

    rows, anomalies = [], []
    has = {c: c in a for c in ("co2", "power_factor", "load_type")}
    for s in shifts:
        g = a[a["shift"] == s["name"]]
        e = g["energy_kwh"].dropna()
        expected = int(_in_shift(np.asarray(gmod), s).sum())
        op_hours = expected * interval_minutes / 60
        mean, std = (e.mean(), e.std(ddof=0)) if len(e) else (np.nan, np.nan)
        hi_t, lo_t = mean + 2 * std, mean - 2 * std
        hi, lo = g[g["energy_kwh"] > hi_t], g[g["energy_kwh"] < lo_t]
        for kind, sub, thr in (("high", hi, hi_t), ("low", lo, lo_t)):
            for _, r in sub.iterrows():
                anomalies.append({
                    "timestamp": str(r["timestamp"]), "shift": s["name"],
                    "value_kwh": _r(r["energy_kwh"]), "type": kind, "threshold_kwh": _r(thr),
                    "_z": abs(r["energy_kwh"] - mean) / std if std else 0,
                })

        daily = g.groupby(g["interval_start"].dt.date)["energy_kwh"].sum()
        idx = pd.to_datetime(pd.Series(daily.index))
        weekend = (idx.dt.dayofweek >= 5).to_numpy()
        coverage = round(len(e) / expected * 100, 2) if expected else 0.0
        if expected and coverage < 90:
            warnings.append(f"{s['name']} has only {coverage}% of expected readings.")

        row = {
            "name": s["name"],
            "total_kwh": _r(e.sum()),
            "energy_per_operating_hour": _r(e.sum() / op_hours) if op_hours else None,
            "avg_kwh_per_interval": _r(mean),
            "peak_kwh": _r(e.max()) if len(e) else None,
            "peak_time": str(g.loc[e.idxmax(), "timestamp"]) if len(e) else None,
            "min_kwh": _r(e.min()) if len(e) else None,
            "min_time": str(g.loc[e.idxmin(), "timestamp"]) if len(e) else None,
            "std_kwh": _r(std),
            "readings": int(len(e)),
            "expected_readings": expected,
            "coverage_pct": coverage,
            "high_anomalies": int(len(hi)),
            "low_anomalies": int(len(lo)),
            "weekday_avg_daily_kwh": _r(daily[~weekend].mean()) if (~weekend).any() else None,
            "weekend_avg_daily_kwh": _r(daily[weekend].mean()) if weekend.any() else None,
            "total_co2": _r(g["co2"].sum(), 4) if has["co2"] else None,
            "avg_power_factor": _r(g["power_factor"].mean()) if has["power_factor"] else None,
            "load_type_mix": (
                {k: _r(v * 100) for k, v in g["load_type"].value_counts(normalize=True).items()}
                if has["load_type"] and len(g) else None
            ),
        }
        rows.append(row)

    unassigned = int((a["shift"] == "Unassigned").sum())
    if unassigned:
        warnings.append(f"{unassigned} readings fall outside all defined shifts (Unassigned).")

    valid = [r for r in rows if r["total_kwh"] is not None and r["readings"]]
    comp = {}
    if valid:
        by_total = sorted(valid, key=lambda r: r["total_kwh"], reverse=True)
        by_hour = sorted(valid, key=lambda r: r["energy_per_operating_hour"] or 0, reverse=True)
        hi_r, lo_r = by_total[0], by_total[-1]
        comp = {
            "highest_total": hi_r["name"],
            "lowest_total": lo_r["name"],
            "pct_diff_highest_vs_lowest": _r((hi_r["total_kwh"] - lo_r["total_kwh"]) / lo_r["total_kwh"] * 100)
            if lo_r["total_kwh"] else None,
            "ranking_by_total": [r["name"] for r in by_total],
            "ranking_by_per_hour": [r["name"] for r in by_hour],
            "highest_co2": max(valid, key=lambda r: r["total_co2"])["name"] if has["co2"] else None,
            "lowest_avg_power_factor": min(valid, key=lambda r: r["avg_power_factor"])["name"]
            if has["power_factor"] else None,
        }

    anomalies.sort(key=lambda x: x["_z"], reverse=True)
    for x in anomalies:
        x.pop("_z")

    return {
        "shift_definitions": [
            {"name": s["name"], "start": s["start"], "end": s["end"], "duration_hours": _duration_hours(s)}
            for s in shifts
        ],
        "assumption_note": ASSUMPTION_NOTE,
        "data_period": {"start": str(a["timestamp"].min()), "end": str(a["timestamp"].max()), "days": days},
        "shifts": rows,
        "comparisons": comp,
        "anomalies": anomalies[:20],
        "data_quality_warnings": warnings,
    }


def daily_shift_totals(df: pd.DataFrame) -> pd.DataFrame:
    a = df[df["shift"] != "Unassigned"]
    agg = {"total_kwh": ("energy_kwh", "sum")}
    if "co2" in a:
        agg["total_co2"] = ("co2", "sum")
    return (a.groupby([a["interval_start"].dt.date.rename("date"), "shift"]).agg(**agg).reset_index())


def hourly_profile(df: pd.DataFrame) -> pd.DataFrame:
    a = df[df["shift"] != "Unassigned"]
    return (a.groupby([a["interval_start"].dt.hour.rename("hour"), "shift"])["energy_kwh"]
            .mean().rename("avg_kwh").reset_index())


def weekday_hour_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Average kWh per interval by day of week x hour (for the heatmap)."""
    t = df["interval_start"]
    m = df.groupby([t.dt.dayofweek.rename("dow"), t.dt.hour.rename("hour")])["energy_kwh"].mean().unstack()
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    m.index = [names[i] for i in m.index]
    return m
