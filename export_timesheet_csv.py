from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export structured timesheet JSON into easy-to-read CSV files."
    )
    parser.add_argument("json_path", help="Path to the structured timesheet JSON file.")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional output folder. Defaults to the JSON file folder.",
    )
    return parser.parse_args()


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def safe_file_stem(value: str, fallback: str) -> str:
    candidate = clean_text(value)
    if not candidate:
        candidate = fallback
    candidate = re.sub(r"[<>:\"/\\|?*\x00-\x1F]", "_", candidate)
    candidate = re.sub(r"\s+", "_", candidate)
    candidate = candidate.strip(" ._")
    return candidate or fallback


def unique_output_path(output_dir: Path, base_name: str, suffix: str) -> Path:
    candidate = output_dir / f"{base_name}{suffix}"
    counter = 2
    while candidate.exists():
        candidate = output_dir / f"{base_name}_{counter}{suffix}"
        counter += 1
    return candidate


def write_daily_rows_csv(payload: dict, output_path: Path) -> None:
    rows = payload.get("daily_rows", [])
    base_fields = [
        "source_file",
        "employee_name",
        "wp_fin_no",
        "employee_id",
        "month",
        "year",
    ]
    row_fields = [
        "day",
        "regular_hours",
        "ot_hours",
        "rope_access_allowance",
        "transport_allowance",
        "night_shift_allowance",
        "food_allowance",
        "job_site",
        "supervisor_signature",
    ]
    fieldnames = base_fields + row_fields

    header = payload.get("header", {})
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "source_file": payload.get("source_file", ""),
                    "employee_name": header.get("employee_name", ""),
                    "wp_fin_no": header.get("wp_fin_no", ""),
                    "employee_id": header.get("employee_id", ""),
                    "month": header.get("month", ""),
                    "year": header.get("year", ""),
                    "day": row.get("day", ""),
                    "regular_hours": row.get("regular_hours", ""),
                    "ot_hours": row.get("ot_hours", ""),
                    "rope_access_allowance": row.get("rope_access_allowance", ""),
                    "transport_allowance": row.get("transport_allowance", ""),
                    "night_shift_allowance": row.get("night_shift_allowance", ""),
                    "food_allowance": row.get("food_allowance", ""),
                    "job_site": row.get("job_site", ""),
                    "supervisor_signature": row.get("supervisor_signature", ""),
                }
            )


def main() -> None:
    args = parse_args()
    json_path = Path(args.json_path).resolve()
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else json_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    header = payload.get("header", {})
    stem = safe_file_stem(header.get("employee_name", ""), json_path.stem)
    daily_csv = unique_output_path(output_dir, f"{stem}_daily_rows", ".csv")

    write_daily_rows_csv(payload, daily_csv)

    print(f"Daily rows CSV: {daily_csv}")


if __name__ == "__main__":
    main()
