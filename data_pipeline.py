"""Data ingestion and cleaning for the Manufacturing Shift Trend Brief (PS 198).

Public API:
    load_and_clean(file, timestamp_col=None, energy_col=None, unit=None)
        -> (clean_df, quality_report)

Works with the UCI "Steel Industry Energy Consumption" CSV and with any CSV
that has a timestamp column and an energy column.

NOTE: in the UCI data the timestamp marks the END of each 15-min interval
(00:15 = energy used during 00:00-00:15). We keep timestamps as given; shift
logic downstream should assign records to shifts with that in mind.
"""
from __future__ import annotations

import json
import re
import sys
import warnings
from pathlib import Path

import pandas as pd

# Name hints, in priority order (first hint wins, then column order).
TIMESTAMP_HINTS = ["timestamp", "datetime", "date", "time"]
ENERGY_HINTS = ["kwh", "usage", "energy", "consumption", "kw", "power"]
# Columns that look like energy by name but are not (UCI has several of these).
ENERGY_EXCLUDE = ("co2", "factor", "reactive", "kvar", "nsm")

# Known UCI source columns -> output columns. Used first; auto-detection is the fallback.
UCI_COLUMNS = {
    "timestamp": "date",
    "energy_kwh": "Usage_kWh",
    "co2": "CO2(tCO2)",
    "power_factor": "Lagging_Current_Power_Factor",
    "load_type": "Load_Type",
}

UNITS = {"kwh": "kWh", "kw": "kW"}
UCI_DATE_FORMAT = "%d/%m/%Y %H:%M"


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _read_csv(file) -> pd.DataFrame:
    """Read a path or a file-like object (e.g. Streamlit UploadedFile)."""
    if hasattr(file, "seek"):
        file.seek(0)
    # utf-8-sig strips the BOM that the UCI file's header carries.
    df = pd.read_csv(file, encoding="utf-8-sig", encoding_errors="replace")
    df.columns = [str(c).lstrip("\ufeff").strip() for c in df.columns]
    return df


def _resolve_col(df: pd.DataFrame, name: str, label: str) -> str:
    """Match a user-given column name exactly, then case-insensitively."""
    if name in df.columns:
        return name
    lowered = {c.lower(): c for c in df.columns}
    if str(name).strip().lower() in lowered:
        return lowered[str(name).strip().lower()]
    raise ValueError(f"{label} column '{name}' not found. Available columns: {list(df.columns)}")


def _candidates(columns, hints, exclude=()) -> list[str]:
    """Columns whose name contains a hint, ordered by hint priority."""
    found: list[str] = []
    for hint in hints:
        for col in columns:
            low = str(col).lower()
            if hint in low and not any(x in low for x in exclude) and col not in found:
                found.append(col)
    return found


def _parse_timestamps(series: pd.Series) -> pd.Series:
    """Parse to datetime (dayfirst). Unparseable values become NaT."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    s = series.astype(str).str.strip()
    # Fast, strict path for the UCI format "dd/mm/yyyy HH:MM".
    parsed = pd.to_datetime(s, format=UCI_DATE_FORMAT, errors="coerce")
    # Anything left over (other CSVs, ISO dates, seconds, ...) -> flexible parse.
    mask = parsed.isna()
    if mask.any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed.loc[mask] = pd.to_datetime(s[mask], dayfirst=True, errors="coerce")
    return parsed


def _detect_timestamp_col(df: pd.DataFrame) -> str:
    named = _candidates(df.columns, TIMESTAMP_HINTS)
    others = [c for c in df.columns if c not in named and not pd.api.types.is_numeric_dtype(df[c])]
    # Name-matched columns need >=50% parseable; unnamed guesses need >=80%.
    for col, min_ok in [(c, 0.5) for c in named] + [(c, 0.8) for c in others]:
        if _parse_timestamps(df[col]).notna().mean() >= min_ok:
            return col
    raise ValueError(f"Could not auto-detect a timestamp column. Columns: {list(df.columns)}. "
                     "Pass timestamp_col=...")


def _detect_energy_col(df: pd.DataFrame, ts_col: str) -> str:
    others = [c for c in df.columns if c != ts_col]
    for col in _candidates(others, ENERGY_HINTS, ENERGY_EXCLUDE):
        if pd.to_numeric(df[col], errors="coerce").notna().mean() > 0.5:
            return col
    raise ValueError(f"Could not auto-detect an energy column. Columns: {list(df.columns)}. "
                     "Pass energy_col=...")


def _detect_unit(col_name: str) -> str | None:
    """Guess kW vs kWh from the column name. None if the name gives no clue."""
    low = col_name.lower()
    if "kwh" in low:
        return "kWh"
    if re.search(r"(?<![a-z])kw(?![a-z])", low):
        return "kW"
    return None


def _infer_interval_minutes(ts: pd.Series) -> float | None:
    """Mode of positive time differences between consecutive timestamps."""
    diffs = ts.drop_duplicates().sort_values().diff().dropna()
    diffs = diffs[diffs > pd.Timedelta(0)]
    if diffs.empty:
        return None
    return diffs.mode().iloc[0].total_seconds() / 60


def _num(x: float) -> int | float:
    """12.0 -> 12 so the report reads cleanly."""
    return int(x) if float(x).is_integer() else round(float(x), 3)


def _fmt(t: pd.Timestamp) -> str:
    return t.isoformat(sep=" ")


def _pick_optional_cols(raw: pd.DataFrame, used: set[str]) -> dict[str, str]:
    """Map output name -> source column for the optional co2 / power_factor / load_type."""
    cols = [c for c in raw.columns if c not in used]
    out: dict[str, str] = {}
    for name in ("co2", "power_factor", "load_type"):
        if UCI_COLUMNS[name] in cols:  # exact UCI header
            out[name] = UCI_COLUMNS[name]

    # Fallbacks for other CSVs
    if "co2" not in out:
        co2 = [c for c in cols if "co2" in c.lower()]
        if co2:
            out["co2"] = co2[0]
    if "power_factor" not in out:
        pf = [c for c in cols if re.search(r"power[_ .]?factor", c.lower())]
        if pf:  # prefer lagging if there are several
            out["power_factor"] = next((c for c in pf if "lagging" in c.lower()), pf[0])
    if "load_type" not in out:
        lt = [c for c in cols if re.sub(r"[_ ]", "", c.lower()) == "loadtype"]
        if lt:
            out["load_type"] = lt[0]
    return out


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def load_and_clean(file, timestamp_col: str | None = None, energy_col: str | None = None,
                   unit: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Load a CSV (path or Streamlit UploadedFile) and return (clean_df, quality_report).

    clean_df columns: timestamp, energy_kwh, plus co2 / power_factor / load_type if present.
    `unit` ("kW" or "kWh") overrides auto-detection; `unit_detected` in the report is the
    unit that was actually applied.
    """
    warns: list[str] = []
    raw = _read_csv(file)

    # 1. Column selection (manual override > auto-detect)
    if timestamp_col:
        ts_col = _resolve_col(raw, timestamp_col, "Timestamp")
    elif UCI_COLUMNS["timestamp"] in raw.columns:
        ts_col = UCI_COLUMNS["timestamp"]
    else:
        ts_col = _detect_timestamp_col(raw)

    if energy_col:
        en_col = _resolve_col(raw, energy_col, "Energy")
    elif UCI_COLUMNS["energy_kwh"] in raw.columns:
        en_col = UCI_COLUMNS["energy_kwh"]
    else:
        en_col = _detect_energy_col(raw, ts_col)
    if ts_col == en_col:
        raise ValueError("Timestamp and energy columns must be different.")

    # 2. Parse timestamps and energy; drop rows with null timestamps
    timestamps = _parse_timestamps(raw[ts_col])
    energy = pd.to_numeric(raw[en_col], errors="coerce")
    non_numeric = int((raw[en_col].notna() & energy.isna()).sum())
    if non_numeric:
        warns.append(f"{non_numeric} non-numeric energy value(s) in '{en_col}' were set to NaN.")

    df = pd.DataFrame({"timestamp": timestamps, "energy_kwh": energy})
    optional = _pick_optional_cols(raw, {ts_col, en_col})
    for out_name, src in optional.items():
        if out_name == "load_type":
            df[out_name] = raw[src].astype(str).str.strip().where(raw[src].notna())  # NaN stays NaN
        else:
            df[out_name] = pd.to_numeric(raw[src], errors="coerce")

    bad_ts = int(df["timestamp"].isna().sum())
    if bad_ts:
        warns.append(f"Dropped {bad_ts} row(s) with missing/unparseable timestamps.")
        df = df.dropna(subset=["timestamp"])
    if df.empty:
        raise ValueError("No rows with valid timestamps were found.")

    # 3. Sort and drop exact duplicate records
    df = df.sort_values("timestamp", kind="mergesort")
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    duplicates_removed = before - len(df)

    repeated_ts = int(df["timestamp"].duplicated().sum())
    if repeated_ts:
        warns.append(f"{repeated_ts} row(s) share a timestamp with a different value; kept as-is.")

    # 4. Interval
    interval = _infer_interval_minutes(df["timestamp"])
    if interval is None:
        warns.append("Could not infer the sampling interval (need at least 2 distinct timestamps).")

    # 5. Unit detection / conversion
    detected = _detect_unit(en_col)
    if unit is not None:
        if str(unit).strip().lower() not in UNITS:
            raise ValueError("unit must be 'kW' or 'kWh'.")
        unit_used = UNITS[str(unit).strip().lower()]
        if detected and detected != unit_used:
            warns.append(f"Column name suggests {detected}, but unit override '{unit_used}' was applied.")
    elif detected:
        unit_used = detected
    else:
        unit_used = "kWh"
        warns.append(f"Could not tell the unit from column '{en_col}'; assumed kWh.")

    if unit_used == "kW":
        if interval is None:
            warns.append("Unit is kW but the interval is unknown; values were NOT converted to kWh.")
        else:
            df["energy_kwh"] = df["energy_kwh"] * interval / 60
    df["energy_kwh"] = df["energy_kwh"].astype(float)

    # 6. Missing energy (kept as NaN, only counted)
    missing_energy = int(df["energy_kwh"].isna().sum())
    if missing_energy:
        warns.append(f"{missing_energy} row(s) have missing energy (left as NaN).")
    for col in ("co2", "power_factor", "load_type"):
        if col in df.columns and df[col].isna().any():
            warns.append(f"{int(df[col].isna().sum())} row(s) have missing '{col}' (left as NaN).")
    if "power_factor" in df.columns and ((df["power_factor"] < 0) | (df["power_factor"] > 100)).any():
        warns.append("power_factor has values outside the expected 0-100 range.")
    negatives = int((df["energy_kwh"] < 0).sum())
    if negatives:
        warns.append(f"{negatives} row(s) have negative energy values.")

    # 7. Gaps
    uniq = df["timestamp"].drop_duplicates().reset_index(drop=True)
    start, end = uniq.iloc[0], uniq.iloc[-1]
    largest_gaps: list[dict] = []
    if interval is None:
        expected = len(uniq)
    else:
        expected = int(round((end - start).total_seconds() / 60 / interval)) + 1
        diffs = uniq.diff()
        # A gap = distance between consecutive records clearly larger than one interval.
        gap_idx = diffs.index[diffs > pd.Timedelta(minutes=interval * 1.5)]
        gaps = sorted(((diffs[i].total_seconds() / 60, uniq[i - 1], uniq[i]) for i in gap_idx),
                      reverse=True)[:5]
        largest_gaps = [{"start": _fmt(a), "end": _fmt(b), "minutes": _num(m)} for m, a, b in gaps]
    missing_intervals = max(expected - len(uniq), 0)
    coverage = min(len(uniq) / expected * 100, 100.0)
    if missing_intervals:
        warns.append(f"{missing_intervals} expected interval(s) are missing (coverage {coverage:.2f}%).")

    report = {
        "rows": int(len(df)),
        "start": _fmt(start),
        "end": _fmt(end),
        "interval_minutes": _num(interval) if interval is not None else None,
        "unit_detected": unit_used,
        "columns_used": {"timestamp": ts_col, "energy_kwh": en_col, **optional},
        "duplicates_removed": int(duplicates_removed),
        "missing_energy": missing_energy,
        "expected_intervals": int(expected),
        "missing_intervals": int(missing_intervals),
        "coverage_pct": round(float(coverage), 2),
        "largest_gaps": largest_gaps,  # start/end = last record before / first record after the gap
        "warnings": warns,
    }
    return df, report


# --------------------------------------------------------------------------- #
# Demo: python data_pipeline.py [path/to/Steel_industry_data.csv]
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    if len(sys.argv) > 1 and Path(sys.argv[1]).exists():
        source = sys.argv[1]
    else:  # tiny synthetic UCI-style sample (with BOM) so the demo runs without the dataset
        import io

        idx = pd.date_range("2018-01-01 00:15", periods=96 * 2, freq="15min")
        sample = pd.DataFrame({
            "date": idx.strftime("%d/%m/%Y %H:%M"),
            "Usage_kWh": 3.0 + (idx.hour >= 8) * 20.0,
            "CO2(tCO2)": 0.0,
            "Lagging_Current_Power_Factor": 73.5,
            "Load_Type": "Light_Load",
        }).drop(index=range(40, 48))  # simulate a 2-hour outage
        source = io.BytesIO(sample.to_csv(index=False).encode("utf-8-sig"))

    clean, rep = load_and_clean(source)
    print(json.dumps(rep, indent=2))
    print(clean.head(5))
