#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable


def extract_record(json_path: Path, root_dir: Path) -> dict[str, str | float]:
    rel_parts = json_path.relative_to(root_dir).parts
    if len(rel_parts) < 4:
        raise ValueError(f"Unexpected path layout: {json_path}")

    domain = rel_parts[0]
    method = rel_parts[1]

    # Support both:
    #   <domain>/<method>/eval_results/*.json
    #   <domain>/<method>/ALM_ON/eval_results/*.json
    if rel_parts[2] in {"ALM_ON", "ALM_OFF"}:
        alm_dir = rel_parts[2]
        eval_dir_index = 3
    else:
        alm_dir = ""
        eval_dir_index = 2

    if rel_parts[eval_dir_index] != "eval_results":
        raise ValueError(f"Unexpected eval_results layout: {json_path}")

    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    results = payload.get("results", {})
    wmdp_key = f"wmdp_{domain}"
    wmdp_acc = results.get(wmdp_key, {}).get("acc,none")
    mmlu_acc = results.get("mmlu", {}).get("acc,none")

    return {
        "domain": domain,
        "method": method,
        "alm_dir": alm_dir,
        "run_name": json_path.stem,
        "wmdp_task": wmdp_key,
        "wmdp_acc": "" if wmdp_acc is None else float(wmdp_acc),
        "mmlu_acc": "" if mmlu_acc is None else float(mmlu_acc),
        "json_path": str(json_path),
    }


def iter_records(root_dir: Path) -> Iterable[dict[str, str | float]]:
    for json_path in sorted(root_dir.glob("**/eval_results/*.json")):
        try:
            yield extract_record(json_path, root_dir)
        except Exception as exc:
            print(f"[WARN] Skip {json_path}: {exc}")


def sort_records(records: list[dict[str, str | float]], sort_by: str) -> list[dict[str, str | float]]:
    if sort_by in {"wmdp_acc", "mmlu_acc"}:
        return sorted(
            records,
            key=lambda row: (row[sort_by] == "", row[sort_by]),
            reverse=True,
        )
    return sorted(records, key=lambda row: tuple(str(row[key]) for key in ("domain", "method", "alm_dir", "run_name")))


def print_preview(records: list[dict[str, str | float]], limit: int) -> None:
    if not records:
        print("[INFO] No records found.")
        return

    headers = ("domain", "method", "alm_dir", "wmdp_acc", "mmlu_acc", "run_name")
    widths = {header: len(header) for header in headers}
    preview_rows = records[:limit]

    for row in preview_rows:
        for header in headers:
            widths[header] = max(widths[header], len(str(row[header])))

    header_line = "  ".join(header.ljust(widths[header]) for header in headers)
    print(header_line)
    print("  ".join("-" * widths[header] for header in headers))
    for row in preview_rows:
        print("  ".join(str(row[header]).ljust(widths[header]) for header in headers))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize wmdp_{domain} and mmlu accuracy from grid-search eval JSONs.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("models/grid_search-v1"),
        help="Grid-search root directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/grid_search-v1/summary_wmdp_mmlu.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--sort-by",
        choices=("path", "wmdp_acc", "mmlu_acc"),
        default="path",
        help="Sort order for CSV and preview.",
    )
    parser.add_argument(
        "--domain",
        choices=("bio", "cyber", "both"),
        default=None,
        help="Optionally filter to a single domain.",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=20,
        help="Number of rows to print after writing the CSV.",
    )
    args = parser.parse_args()

    records = list(iter_records(args.root))
    if args.domain is not None:
        records = [record for record in records if record["domain"] == args.domain]
    records = sort_records(records, args.sort_by)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["domain", "method", "alm_dir", "wmdp_task", "wmdp_acc", "mmlu_acc", "run_name", "json_path"]
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"[INFO] Wrote {len(records)} rows to {args.output}")
    print_preview(records, args.preview)


if __name__ == "__main__":
    main()
