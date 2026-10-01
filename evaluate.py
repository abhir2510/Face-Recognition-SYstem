"""Calculate basic recognition metrics from a CSV of manual test outcomes."""
from __future__ import annotations

import argparse
import csv

from sklearn.metrics import classification_report, confusion_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a manually labelled recognition test CSV.")
    parser.add_argument("csv_file", help="CSV with columns: actual,predicted")
    args = parser.parse_args()
    with open(args.csv_file, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows or not {"actual", "predicted"}.issubset(rows[0]):
        raise ValueError("CSV must contain actual,predicted columns.")
    actual, predicted = [row["actual"] for row in rows], [row["predicted"] for row in rows]
    labels = sorted(set(actual) | set(predicted))
    print(classification_report(actual, predicted, labels=labels, zero_division=0))
    print("Labels:", labels)
    print("Confusion matrix:\n", confusion_matrix(actual, predicted, labels=labels))


if __name__ == "__main__":
    main()
