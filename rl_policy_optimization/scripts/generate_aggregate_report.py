from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate plots and markdown report from aggregate_summary.json")
    parser.add_argument("--input", default="outputs/aggregate_summary.json")
    parser.add_argument("--output-dir", default="outputs/report")
    parser.add_argument("--title", default="RL Benchmark Aggregate Report")
    return parser.parse_args()


def load_data(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Aggregate summary must be a list.")
    return [item for item in payload if isinstance(item, dict) and "algo" in item]


def make_bar_plot(labels: list[str], values: list[float], title: str, xlabel: str, out_file: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(labels, values)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.invert_yaxis()

    for bar, value in zip(bars, values):
        ax.text(value, bar.get_y() + bar.get_height() / 2, f" {value:.2f}", va="center")

    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def make_scatter(perf: list[float], speed: list[float], labels: list[str], out_file: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(speed, perf)
    ax.set_title("Performance vs Time")
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("Max Avg Reward (100)")

    for x, y, label in zip(speed, perf, labels):
        ax.annotate(label, (x, y), xytext=(5, 3), textcoords="offset points", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def format_table(rows: list[dict]) -> str:
    header = "| Rank | Algo | Runs | MaxAvg100 | FinalAvg100 | Time(s) | Efficiency |\n"
    sep = "|---:|---|---:|---:|---:|---:|---:|\n"
    lines = [header, sep]
    for idx, row in enumerate(rows, start=1):
        lines.append(
            "| {rank} | {algo} | {runs} | {maxv:.2f} | {finalv:.2f} | {timev:.2f} | {eff:.3f} |\n".format(
                rank=idx,
                algo=row["algo"],
                runs=row.get("runs", 0),
                maxv=row.get("max_avg_reward_100_mean", 0.0),
                finalv=row.get("final_avg_reward_100_mean", 0.0),
                timev=row.get("elapsed_sec_mean", 0.0),
                eff=row.get("efficiency", 0.0),
            )
        )
    return "".join(lines)


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    rows = load_data(input_path)
    if not rows:
        raise RuntimeError("No rows found in aggregate summary.")

    for row in rows:
        elapsed = float(row.get("elapsed_sec_mean", 0.0) or 0.0)
        max_avg = float(row.get("max_avg_reward_100_mean", 0.0) or 0.0)
        row["efficiency"] = (max_avg / elapsed) if elapsed > 0 else 0.0

    by_perf = sorted(rows, key=lambda r: r.get("max_avg_reward_100_mean", 0.0), reverse=True)
    by_final = sorted(rows, key=lambda r: r.get("final_avg_reward_100_mean", 0.0), reverse=True)
    by_time = sorted(rows, key=lambda r: r.get("elapsed_sec_mean", 0.0))

    perf_labels = [r["algo"] for r in by_perf]
    perf_vals = [float(r.get("max_avg_reward_100_mean", 0.0)) for r in by_perf]
    final_vals = [float(r.get("final_avg_reward_100_mean", 0.0)) for r in by_perf]
    time_labels = [r["algo"] for r in by_time]
    time_vals = [float(r.get("elapsed_sec_mean", 0.0)) for r in by_time]

    max_plot = plots_dir / "max_avg_reward_100.png"
    final_plot = plots_dir / "final_avg_reward_100.png"
    time_plot = plots_dir / "elapsed_seconds.png"
    scatter_plot = plots_dir / "performance_vs_time.png"

    make_bar_plot(perf_labels, perf_vals, "Max Average Reward (window=100)", "Reward", max_plot)
    make_bar_plot(perf_labels, final_vals, "Final Average Reward (window=100)", "Reward", final_plot)
    make_bar_plot(time_labels, time_vals, "Elapsed Time", "Seconds", time_plot)
    make_scatter(perf_vals, [float(r.get("elapsed_sec_mean", 0.0)) for r in by_perf], perf_labels, scatter_plot)

    top3 = by_perf[:3]
    low3 = by_perf[-3:]
    times = [float(r.get("elapsed_sec_mean", 0.0)) for r in rows]
    median_time = statistics.median(times)

    report_md = output_dir / "aggregate_report.md"
    report_md.write_text(
        "\n".join(
            [
                f"# {args.title}",
                "",
                f"Source data: `{input_path.as_posix()}`",
                f"Total algorithms: **{len(rows)}**",
                "",
                "## Leaderboard (by MaxAvg100)",
                "",
                format_table(by_perf),
                "",
                "## Plots",
                "",
                f"![MaxAvg100](plots/{max_plot.name})",
                "",
                f"![FinalAvg100](plots/{final_plot.name})",
                "",
                f"![ElapsedSeconds](plots/{time_plot.name})",
                "",
                f"![PerformanceVsTime](plots/{scatter_plot.name})",
                "",
                "## Key observations",
                "",
                "- Top-3 by `max_avg_reward_100_mean`: "
                + ", ".join([f"**{r['algo']}** ({r.get('max_avg_reward_100_mean', 0.0):.2f})" for r in top3]),
                "- Bottom-3 by `max_avg_reward_100_mean`: "
                + ", ".join([f"**{r['algo']}** ({r.get('max_avg_reward_100_mean', 0.0):.2f})" for r in low3]),
                "- Fastest method: "
                + f"**{by_time[0]['algo']}** ({by_time[0].get('elapsed_sec_mean', 0.0):.2f}s)",
                "- Slowest method: "
                + f"**{by_time[-1]['algo']}** ({by_time[-1].get('elapsed_sec_mean', 0.0):.2f}s)",
                f"- Median elapsed time across methods: **{median_time:.2f}s**",
                "",
                "## Reproducibility note",
                "",
                "Current summary appears to use single-run statistics (`runs=1` for each method), "
                "so standard deviations are zero. For robust comparisons, aggregate multiple seeds.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Report: {report_md}")
    print(f"Plots:  {plots_dir}")


if __name__ == "__main__":
    main()
