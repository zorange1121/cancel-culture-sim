"""Rebuild mean cascade trajectories from existing raw CSVs without simulation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from run_kappa_sweep_multiseed import (
    DEFAULT_OUTPUT_DIR,
    aggregate_forward_filled_trajectories,
    plot_trajectories,
    write_csv,
)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def normalize_timestep_rows(rows: list[dict]) -> list[dict]:
    for row in rows:
        row["kappa"] = float(row["kappa"])
        row["random_seed"] = int(row["random_seed"])
        row["timestep"] = int(row["timestep"])
        row["activated_count"] = int(row["activated_count"])
        row["activated_proportion"] = float(row["activated_proportion"])
    return rows


def normalize_run_rows(rows: list[dict]) -> list[dict]:
    for row in rows:
        row["kappa"] = float(row["kappa"])
        row["random_seed"] = int(row["random_seed"])
        row["final_activated_proportion"] = float(row["final_activated_proportion"])
        row["final_activated_count"] = int(row["final_activated_count"])
        row["cascade_duration"] = int(row["cascade_duration"])
        row["t50"] = int(row["t50"])
    return rows


def main() -> None:
    output_dir = DEFAULT_OUTPUT_DIR
    timestep_path = output_dir / "raw_timestep_results.csv"
    run_path = output_dir / "raw_run_results.csv"
    config_path = output_dir / "experiment_config.json"
    aggregate_path = output_dir / "mean_trajectory_forward_filled.csv"
    canonical_figure_path = output_dir / "figure3_mean_cascade_trajectories.png"
    explicit_figure_path = (
        output_dir / "figure3_mean_cascade_trajectories_forward_filled.png"
    )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    horizon = int(config["max_steps"])
    timestep_rows = normalize_timestep_rows(read_csv(timestep_path))
    run_rows = normalize_run_rows(read_csv(run_path))
    aggregate_rows = aggregate_forward_filled_trajectories(
        timestep_rows, run_rows, horizon=horizon
    )

    write_csv(aggregate_path, aggregate_rows)
    plot_trajectories(aggregate_rows, canonical_figure_path)
    plot_trajectories(aggregate_rows, explicit_figure_path)

    expected_rows = 4 * 11 * (horizon + 1)
    if len(aggregate_rows) != expected_rows:
        raise AssertionError(
            f"Expected {expected_rows} aggregate rows, found {len(aggregate_rows)}"
        )
    if {row["n_runs"] for row in aggregate_rows} != {20}:
        raise AssertionError("Every aggregate timestep must contain all 20 runs")

    print(f"Read {len(run_rows)} existing runs; no simulation was executed")
    print(f"Wrote {len(aggregate_rows)} aggregate rows through timestep {horizon}")
    print(f"Aggregate CSV: {aggregate_path.resolve()}")
    print(f"Corrected figure: {explicit_figure_path.resolve()}")


if __name__ == "__main__":
    main()
