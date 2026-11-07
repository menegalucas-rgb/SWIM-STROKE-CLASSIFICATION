"""Export the approved/reviewed 1002 25-meter-repetition labels.

Usage:
    python export_1002_approved_labels.py

Place this script beside 1002_LABELED_DATA.csv. No third-party packages
are needed. The original file is never overwritten.

The boundary table below reproduces the reviewed 1002 labels,
including the four screenshot-reviewed corrections (reps 13, 15, 16, 20).
Boundaries are aligned to exact recorded sample timestamps to avoid rounding
classification errors at boundary-adjacent rows.
All other boundaries remain previously estimated and should be reviewed.
"""

import csv
from io import StringIO
from pathlib import Path

FOLDER = Path(__file__).resolve().parent
INPUT_CSV = FOLDER / "1002_LABELED_DATA.csv"
OUTPUT_CSV = FOLDER / "1002_25M_APPROVED_LABELS.csv"
BOUNDARY_EXPORT = FOLDER / "1002_25M_APPROVED_BOUNDARIES.csv"
LABEL_COLUMN = "label (all lowercase)"

# Fields: length, stroke, scheduled-slot-start, transition-start,
# swimming-start, swimming-end, scheduled-slot-end, provenance columns.
# These are the same saved values from the previously reviewed output.
BOUNDARY_DATA = """length,stroke,slot_start_s,prep_start_s,swim_start_s,swim_end_s,slot_end_s,prep_source,start_source,end_source
1,butterfly,0.000000,0.509145737,4.077617168,16.15834904,42.000000,auto-context,auto-sustained-butterfly,auto-accel-last-peak-3s
2,backstroke,42.000000,44.97349381,49.41879034,64.34924293,85.000000,auto-context,auto-sustained-backstroke,auto-accel-last-peak-3s
3,breaststroke,85.000000,87.25295997,90.70145512,109.2295954,127.000000,auto-context,auto-sustained-breaststroke,auto-accel-last-peak-3s
4,freestyle,127.000000,128.9835286,135.2924132,150.7107968,173.000000,auto-context,auto-sustained-freestyle,auto-accel-last-peak-3s
5,butterfly,173.000000,177.8697073,183.0622854,195.0221398,216.000000,auto-context,auto-sustained-butterfly,auto-accel-last-peak-3s
6,backstroke,216.000000,218.3438253,221.4932387,236.7120461,259.000000,auto-context,auto-sustained-backstroke,auto-accel-last-peak-3s
7,breaststroke,259.000000,257.1333342,260.8209326,279.8868949,296.000000,auto-context,auto-sustained-breaststroke,auto-accel-last-peak-3s
8,freestyle,296.000000,298.0657883,300.3979435,315.5071075,336.000000,auto-context,auto-sustained-freestyle,auto-accel-last-peak-3s
9,butterfly,336.000000,337.991456,342.5660737,355.0142033,376.000000,auto-context,auto-sustained-butterfly,auto-accel-last-peak-3s
10,backstroke,376.000000,377.9270916,381.0366235,397.1623278,416.000000,auto-context,auto-sustained-backstroke,auto-accel-last-peak-3s
11,breaststroke,416.000000,417.8527136,421.2213795,439.9882364,456.000000,auto-context,auto-sustained-breaststroke,auto-accel-last-peak-3s
12,freestyle,456.000000,457.5990067,460.2201903,475.7978218,496.000000,auto-context,auto-sustained-freestyle,auto-accel-last-peak-3s
13,butterfly,496.000000,498.9200933,503.365145,515.9428287,536.000000,provisional-user-screenshot,provisional-user-screenshot,auto-accel-last-peak-3s
14,backstroke,536.000000,537.5400856,541.0084023,555.519506,576.000000,auto-context,auto-sustained-backstroke,auto-accel-last-peak-3s
15,breaststroke,576.000000,577.465606,582.0401978,600.6375859,616.000000,provisional-user-screenshot,provisional-user-screenshot,auto-accel-last-peak-3s
16,freestyle,616.000000,617.5606191,620.7598507,636.4570038,656.000000,provisional-user-screenshot,provisional-user-screenshot,auto-accel-last-peak-3s
17,butterfly,656.000000,657.6655567,661.3431661,675.1167524,696.000000,auto-context,auto-sustained-butterfly,auto-accel-last-peak-3s
18,backstroke,696.000000,697.8601272,700.7902546,716.5172579,736.000000,auto-context,auto-sustained-backstroke,auto-accel-last-peak-3s
19,breaststroke,736.000000,736.788949,741.2838089,759.7217267,776.000000,auto-context,auto-sustained-breaststroke,auto-accel-last-peak-3s
20,freestyle,776.000000,777.4420717,780.2625713,795.9497533,816.000000,provisional-user-screenshot,provisional-user-screenshot,auto-accel-last-peak-3s
"""


def load_boundaries():
    rows = list(csv.DictReader(StringIO(BOUNDARY_DATA.strip())))
    assert len(rows) == 20
    for row in rows:
        for name in ("slot_start_s", "prep_start_s", "swim_start_s", "swim_end_s", "slot_end_s"):
            row[name] = float(row[name])
        if not (row["prep_start_s"] <= row["swim_start_s"] < row["swim_end_s"]):
            raise ValueError(f"Invalid boundaries for repetition {row['length']}")
    return rows


def label_at(time_seconds, boundaries):
    # Preserve unknown after the documented final scheduled interval.
    if time_seconds >= boundaries[-1]["slot_end_s"]:
        return "unknown"
    # Activity outside marked transition/swim intervals is not_swimming.
    label = "not_swimming"
    for row in boundaries:
        if row["prep_start_s"] <= time_seconds < row["swim_start_s"]:
            label = "transition"
        elif row["swim_start_s"] <= time_seconds < row["swim_end_s"]:
            label = row["stroke"]
    return label


def main():
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(f"Place the original CSV beside this script: {INPUT_CSV.name}")
    boundaries = load_boundaries()
    counts = {}
    with INPUT_CSV.open("r", newline="", encoding="utf-8-sig") as source, \
         OUTPUT_CSV.open("w", newline="", encoding="utf-8") as destination:
        reader = csv.DictReader(source)
        if "seconds_elapsed" not in (reader.fieldnames or []):
            raise ValueError("Input CSV must contain a seconds_elapsed column")
        columns = list(reader.fieldnames)
        if LABEL_COLUMN not in columns:
            columns.append(LABEL_COLUMN)
        writer = csv.DictWriter(destination, fieldnames=columns)
        writer.writeheader()
        for record in reader:
            time_s = float(record["seconds_elapsed"])
            label = label_at(time_s, boundaries)
            record[LABEL_COLUMN] = label
            writer.writerow(record)
            counts[label] = counts.get(label, 0) + 1

    # Export the identical numeric boundaries for editing and documentation.
    with BOUNDARY_EXPORT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(boundaries[0]))
        writer.writeheader()
        for row in boundaries:
            writer.writerow({key: (f"{value:.12f}" if isinstance(value, float) else value)
                             for key, value in row.items()})
    print(f"Saved: {OUTPUT_CSV}")
    print(f"Saved: {BOUNDARY_EXPORT}")
    for label, count in sorted(counts.items()):
        print(f"{label:15s} {count:>7,d}")
    print(f"Total: {sum(counts.values()):,d}")


if __name__ == "__main__":
    main()
