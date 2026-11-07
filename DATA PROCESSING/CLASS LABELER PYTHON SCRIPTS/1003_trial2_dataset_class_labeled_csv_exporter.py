"""Export Trial 2 labeled Apple Watch data using the approved stroke-specific starts.

Run with: python export_trial2_new_labels.py
Requires: python -m pip install pandas numpy

The embedded start times come from the previously generated stroke-specific
boundary file. Swim ends and rest ends are PROVISIONAL and need verification.
To update any boundary, edit the BOUNDARIES table below, then rerun.
"""
from pathlib import Path
import csv
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
INPUT_CSV = HERE / "1003_TRIAL_2_LABELED_DATA.csv"
OUTPUT_CSV = HERE / "1003_TRIAL_2_NEW_LABELS.csv"
BOUNDARY_OUTPUT_CSV = HERE / "trial2_exported_boundaries.csv"

# Each row: (length, stroke, slot_start, swim_start, swim_end, rest_end, slot_end)
# Start times are the stroke-specific detected values from our approved plots.
# End times are currently estimates inherited from the earlier workflow.
BOUNDARIES = [
    (1, 'butterfly', 0.000000, 3.826000, 16.000000, 29.000000, 30.000000),
    (2, 'backstroke', 30.000000, 31.235000, 46.500000, 58.610000, 60.000000),
    (3, 'breaststroke', 60.000000, 62.840000, 80.400000, 88.620000, 90.000000),
    (4, 'freestyle', 90.000000, 92.372000, 105.000000, 118.700000, 120.000000),
    (5, 'butterfly', 120.000000, 122.891000, 136.000000, 149.000000, 150.000000),
    (6, 'backstroke', 150.000000, 150.758000, 166.500000, 178.610000, 180.000000),
    (7, 'breaststroke', 180.000000, 182.493000, 200.400000, 208.620000, 210.000000),
    (8, 'freestyle', 210.000000, 211.496000, 225.000000, 238.700000, 240.000000),
    (9, 'butterfly', 240.000000, 242.413000, 256.000000, 269.000000, 270.000000),
    (10, 'backstroke', 270.000000, 272.075000, 286.500000, 298.610000, 300.000000),
    (11, 'breaststroke', 300.000000, 302.115000, 320.400000, 328.620000, 330.000000),
    (12, 'freestyle', 330.000000, 332.793000, 345.000000, 358.700000, 360.000000),
    (13, 'butterfly', 360.000000, 362.225000, 376.000000, 389.000000, 390.000000),
    (14, 'backstroke', 390.000000, 391.487000, 406.500000, 418.610000, 420.000000),
    (15, 'breaststroke', 420.000000, 422.743000, 440.400000, 448.620000, 450.000000),
    (16, 'freestyle', 450.000000, 452.255000, 465.000000, 478.700000, 480.000000),
    (17, 'butterfly', 480.000000, 481.927000, 496.000000, 509.000000, 510.000000),
    (18, 'backstroke', 510.000000, 510.881000, 526.500000, 538.610000, 540.000000),
    (19, 'breaststroke', 540.000000, 543.034000, 560.400000, 568.620000, 570.000000),
    (20, 'freestyle', 570.000000, 572.266000, 585.000000, 598.700000, 600.000000),
]


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Place {INPUT_CSV.name} beside this script.")

    # Read as text to preserve the original long Apple Watch timestamps exactly.
    df = pd.read_csv(INPUT_CSV, dtype=str, keep_default_na=False)
    if "seconds_elapsed" not in df.columns:
        raise ValueError("CSV needs a seconds_elapsed column")
    seconds = pd.to_numeric(df["seconds_elapsed"].str.strip().str.lstrip("'"), errors="coerce").to_numpy()
    if not np.isfinite(seconds).all():
        raise ValueError("Some seconds_elapsed values are missing or invalid")
    if len(BOUNDARIES) != 20:
        raise ValueError("Expected 20 swimming lengths")

    labels = np.full(len(df), "unknown", dtype=object)
    records = []
    for length, stroke, slot, start, end, rest_end, slot_end in BOUNDARIES:
        if not (slot <= start < end <= rest_end <= slot_end):
            raise ValueError(f"Invalid boundaries for length {length}")
        # Half-open ranges ensure no sample receives two labels.
        labels[(seconds >= slot) & (seconds < slot_end)] = "transition"
        labels[(seconds >= start) & (seconds < end)] = stroke
        labels[(seconds >= end) & (seconds < rest_end)] = "not-swimming"
        records.append({
            "length": length, "stroke": stroke, "slot_start_s": slot,
            "swim_start_s": start, "swim_end_s": end,
            "rest_end_s": rest_end, "slot_end_s": slot_end,
            "start_source": "stroke-specific-detection-reviewed",
            "end_source": "provisional-template",
        })

    df["label (all lowercase)"] = labels
    df.to_csv(OUTPUT_CSV, index=False)
    pd.DataFrame(records).to_csv(BOUNDARY_OUTPUT_CSV, index=False)
    print(f"Saved: {OUTPUT_CSV} ({len(df):,} rows)")
    print(f"Saved boundaries: {BOUNDARY_OUTPUT_CSV}")
    print(df["label (all lowercase)"].value_counts().to_string())
    print("NOTE: Swimming end/rest boundaries are still provisional.")


if __name__ == "__main__":
    main()
