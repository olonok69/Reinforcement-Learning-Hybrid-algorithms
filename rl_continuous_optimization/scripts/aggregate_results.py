from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import typing as tt


METRICS = ("episodes", "elapsed_sec", "max_avg_reward_100", "final_avg_reward_100")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate benchmark result JSON files into mean/std summaries by algorithm."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help=(
            "Input paths: files, directories, or glob patterns. "
            "Example: outputs/*.json runs/seed*/comparison_results.json"
        ),
    )
    parser.add_argument(
        "--output-json",
        default="outputs/aggregate_summary.json",
        help="Path to aggregated JSON output.",
    )
    parser.add_argument(
        "--output-csv",
        default="outputs/aggregate_summary.csv",
        help="Path to aggregated CSV output.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any input path does not resolve to files.",
    )
    return parser.parse_args()


def resolve_input_files(input_args: list[str], strict: bool) -> list[Path]:
    files: list[Path] = []
    missing: list[str] = []

    for item in input_args:
        p = Path(item)

        if any(ch in item for ch in "*?[]"):
            matched = [Path(m) for m in sorted(Path().glob(item))]
            if matched:
                files.extend([m for m in matched if m.is_file()])
            else:
                missing.append(item)
            continue

        if p.is_file():
            files.append(p)
            continue

        if p.is_dir():
            files.extend(sorted(p.glob("*.json")))
            continue

        missing.append(item)

    unique_files = sorted(set(files))
    if strict and missing:
        raise FileNotFoundError(f"Unresolved inputs: {missing}")

    if not unique_files:
        raise FileNotFoundError("No input JSON files found.")

    return unique_files


def load_records(json_file: Path) -> list[dict]:
    payload = json.loads(json_file.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list in {json_file}, got {type(payload).__name__}")
    rows: list[dict] = []
    for item in payload:
        if isinstance(item, dict) and "algo" in item:
            rows.append(item)
    return rows


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.pstdev(values))


def aggregate(rows: list[dict]) -> list[dict]:
    grouped: dict[str, dict[str, list[float]]] = {}

    for row in rows:
        algo = str(row.get("algo", "unknown"))
        if algo not in grouped:
            grouped[algo] = {metric: [] for metric in METRICS}

        for metric in METRICS:
            value = row.get(metric)
            if isinstance(value, (int, float)):
                grouped[algo][metric].append(float(value))

    summary: list[dict] = []
    for algo, metric_map in grouped.items():
        out: dict[str, tt.Any] = {"algo": algo}
        runs = max((len(v) for v in metric_map.values()), default=0)
        out["runs"] = runs

        for metric in METRICS:
            m, s = mean_std(metric_map[metric])
            out[f"{metric}_mean"] = m
            out[f"{metric}_std"] = s

        summary.append(out)

    summary.sort(key=lambda x: x.get("max_avg_reward_100_mean", 0.0), reverse=True)
    return summary


def write_json(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_csv(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not data:
        path.write_text("", encoding="utf-8")
        return

    fields = ["algo", "runs"]
    for metric in METRICS:
        fields.append(f"{metric}_mean")
        fields.append(f"{metric}_std")

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in data:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    files = resolve_input_files(args.inputs, strict=args.strict)

    all_rows: list[dict] = []
    for file_path in files:
        try:
            rows = load_records(file_path)
            all_rows.extend(rows)
        except Exception as exc:
            print(f"[WARN] Skipping {file_path}: {exc}")

    summary = aggregate(all_rows)

    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    write_json(output_json, summary)
    write_csv(output_csv, summary)

    print(f"Loaded files: {len(files)}")
    print(f"Aggregated rows: {len(all_rows)}")
    print(f"Algorithms summarized: {len(summary)}")
    print(f"JSON summary: {output_json}")
    print(f"CSV summary:  {output_csv}")


if __name__ == "__main__":
    main()
