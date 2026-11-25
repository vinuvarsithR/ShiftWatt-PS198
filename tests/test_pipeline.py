import io
import pandas as pd
from data_pipeline import load_and_clean
from shift_engine import assign_shifts, compute_shift_stats, validate_shifts, DEFAULT_SHIFTS
from llm_notes import verify_notes, template_notes


def csv(text):
    return io.BytesIO(("﻿" + text).encode())


def test_bom_dayfirst_and_columns():
    df, r = load_and_clean(csv("date,Usage_kWh,CO2(tCO2),Lagging_Current_Power_Factor,Load_Type\n"
                               "02/01/2018 00:15,3,0,70,Light_Load\n02/01/2018 00:30,4,0,71,Light_Load\n"))
    assert list(df.columns) == ["timestamp", "energy_kwh", "co2", "power_factor", "load_type"]
    assert df["timestamp"].iloc[0].month == 1 and df["timestamp"].iloc[0].day == 2
    assert r["interval_minutes"] == 15


def test_kw_conversion_and_duplicates_and_gaps():
    df, r = load_and_clean(csv("time,power_kw\n2024-01-01 00:00,40\n2024-01-01 00:15,40\n"
                               "2024-01-01 00:15,40\n2024-01-01 01:00,40\n"))
    assert r["unit_detected"] == "kW" and df["energy_kwh"].iloc[0] == 10
    assert r["duplicates_removed"] == 1 and r["missing_intervals"] == 2


def _day():
    ts = pd.date_range("2024-01-01 00:15", periods=96, freq="15min")
    return pd.DataFrame({"timestamp": ts, "energy_kwh": 1.0})


def test_midnight_shift_and_0600_boundary():
    a = assign_shifts(_day(), DEFAULT_SHIFTS, 15, label_at_end=True)
    assert a.loc[a["timestamp"] == "2024-01-01 06:00", "shift"].item() == "Shift C"
    assert a.loc[a["timestamp"] == "2024-01-01 06:15", "shift"].item() == "Shift A"
    assert a.loc[a["timestamp"] == "2024-01-01 23:00", "shift"].item() == "Shift C"
    assert (a["shift"].value_counts() == 32).all()


def test_validate_overlap_and_gap():
    w = validate_shifts([{"name": "A", "start": "06:00", "end": "15:00"},
                         {"name": "B", "start": "14:00", "end": "20:00"}])
    assert any("overlap" in x for x in w) and any("not covered" in x for x in w)


def test_stats_and_template_verify():
    f = compute_shift_stats(assign_shifts(_day(), DEFAULT_SHIFTS), DEFAULT_SHIFTS)
    assert [s["total_kwh"] for s in f["shifts"]] == [32.0, 32.0, 32.0]
    assert [s["energy_per_operating_hour"] for s in f["shifts"]] == [4.0, 4.0, 4.0]
    n = template_notes(f)
    assert verify_notes(n, f)["all_verified"]


def test_verifier_catches_bad_number_and_causal_words():
    f = {"shift_definitions": [{"name": "Shift A", "start": "06:00", "end": "14:00", "duration_hours": 8}],
         "shifts": [{"name": "Shift A", "total_kwh": 421278.64, "avg_power_factor": 83.91, "load_type_mix": {"Light_Load": 48.12}}],
         "comparisons": {"pct_diff_highest_vs_lowest": 589.73}}
    good = {"summary": "Shift A used 421,278.64 kWh (589.7% more), PF 83.9%, light load 48.1%.", "observations": []}
    assert verify_notes(good, f)["all_verified"]
    bad = {"summary": "Shift A used 500,000 kWh because the furnace ran longer.", "observations": []}
    v = verify_notes(bad, f)
    assert v["mismatches"] and v["flagged_phrases"]


def test_alerts_flag_spike_and_drop():
    from alerts import score_readings, detect_alerts
    ts = pd.date_range("2024-01-01 00:15", periods=96 * 28, freq="15min")
    df = pd.DataFrame({"timestamp": ts, "energy_kwh": 20.0})
    df.loc[df.index % 7 == 0, "energy_kwh"] = 21.0  # small natural variation
    df.loc[500, "energy_kwh"] = 120.0  # spike
    df.loc[900, "energy_kwh"] = 1.0  # drop
    ev = detect_alerts(score_readings(assign_shifts(df, DEFAULT_SHIFTS)), 5, 5)
    assert set(ev["direction"]) == {"HIGH", "LOW"} and len(ev) == 2


def test_verifier_catches_number_on_wrong_shift():
    f = {"shift_definitions": [{"name": "Shift A"}, {"name": "Shift B"}],
         "shifts": [{"name": "Shift A", "total_kwh": 421278.64}, {"name": "Shift B", "total_kwh": 470188.13}]}
    ok = {"summary": "Shift A used 421,279 units and Shift B used 470,188 units.", "observations": []}
    assert verify_notes(ok, f)["all_verified"]
    swapped = {"summary": "Shift A used 470,188 units and Shift B used 421,279 units.", "observations": []}
    assert not verify_notes(swapped, f)["all_verified"]
