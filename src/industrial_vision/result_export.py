"""Export helpers for inspection results."""

import csv
import json
from pathlib import Path
from typing import Iterable

from industrial_vision.inspection import InspectionResult


def result_to_summary_row(result: InspectionResult) -> dict[str, object]:
    """Convert an inspection result into one flat CSV-friendly summary row."""

    return {
        "filename": result.filename,
        "status": result.status,
        "passed": result.passed,
        "defect_count": result.defect_count,
        "total_defect_area": round(result.total_defect_area, 3),
        "max_defect_area": round(result.max_defect_area, 3),
    }


def save_inspection_result_json(
    result: InspectionResult,
    output_path: str | Path,
) -> Path:
    """Save one full inspection result as a JSON file."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(result.to_dict(), file, indent=2)

    return path


def save_inspection_summary_csv(
    results: Iterable[InspectionResult],
    output_path: str | Path,
) -> Path:
    """Save one or many inspection results as a summary CSV file."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = [result_to_summary_row(result) for result in results]

    fieldnames = [
        "filename",
        "status",
        "passed",
        "defect_count",
        "total_defect_area",
        "max_defect_area",
    ]

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return path


def save_defect_candidates_csv(
    result: InspectionResult,
    output_path: str | Path,
) -> Path:
    """Save defect candidates from one inspection result as a CSV file."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "filename",
        "candidate_index",
        "x",
        "y",
        "width",
        "height",
        "area",
        "mean_intensity",
    ]

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for index, candidate in enumerate(result.candidates, start=1):
            writer.writerow(
                {
                    "filename": result.filename,
                    "candidate_index": index,
                    "x": candidate.x,
                    "y": candidate.y,
                    "width": candidate.width,
                    "height": candidate.height,
                    "area": round(candidate.area, 3),
                    "mean_intensity": round(candidate.mean_intensity, 3),
                }
            )

    return path
