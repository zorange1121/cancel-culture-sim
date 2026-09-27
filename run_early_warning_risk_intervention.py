"""Held-out early-warning and risk-triggered cascade intervention experiment."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time

import networkx as nx
import numpy as np
import pandas as pd
import scipy
from scipy import stats

from baseline_simulation import K, N, NUM_IGNITE, create_simulation
from cascade_aware import select_cascade_aware_paths
from early_warning import (
    OBSERVATION_TIMESTEP,
    classification_metrics,
    fit_logistic_risk_model,
    observe_baseline_cascade,
    select_f2_threshold,
)
from intervention import InterventionSpec, build_exposure_paths, run_intervention
from network_generation import SUPPORTED_TOPOLOGIES


OUTPUT_DIR = Path("outputs") / "early_warning_risk_intervention_heldout"
TRAIN_SEEDS = tuple(range(0, 50))
VALIDATION_SEEDS = tuple(range(50, 60))
TEST_SEEDS = tuple(range(60, 70))
KAPPAS = (0.4, 0.5, 0.6)
INTERVENTION_KAPPA = 0.5
BUDGETS = (0.02, 0.05)
WEIGHTS = (0.0, 0.75)
GLOBAL_THRESHOLD = 0.5
L2_CANDIDATES = (0.01, 0.1, 1.0, 10.0)
POLICIES = ("fixed_5pct", "all_t5", "risk_t5", "oracle_t5")


def cascade_metrics(counts: np.ndarray) -> dict[str, int | float]:
    final = int(counts[-1])
    return {
        "final_activated_count": final,
        "final_activated_proportion": final / N,
        "cascade_duration": int(counts.size - 1),
        "t50": int(np.flatnonzero(counts >= 0.5 * final)[0]),
    }


def generate_feature_seed(seed: int) -> tuple[list[dict], list[dict]]:
    rows, pairing = [], []
    reference_ignition = None
    split = "train" if seed in TRAIN_SEEDS else "validation" if seed in VALIDATION_SEEDS else "test"
    for topology in SUPPORTED_TOPOLOGIES:
        simulation = create_simulation(
            topology,
            n=N,
            target_avg_degree=K,
            network_seed=seed,
            agent_seed=seed,
        )
        ignition = simulation.rng.choice(simulation.kol_idx, size=NUM_IGNITE, replace=False)
        ignition_tuple = tuple(sorted(map(int, ignition)))
        if reference_ignition is None:
            reference_ignition = ignition_tuple
        pairing.append({
            "topology": topology,
            "random_seed": seed,
            "initial_kol_seed_nodes": json.dumps(ignition_tuple),
            "matches_seed_reference": ignition_tuple == reference_ignition,
        })
        paths = build_exposure_paths(simulation)
        for kappa in KAPPAS:
            features, counts = observe_baseline_cascade(
                simulation,
                kappa,
                ignition,
                observation_timestep=OBSERVATION_TIMESTEP,
                prepared_paths=paths,
            )
            outcome = cascade_metrics(counts)
            rows.append({
                "topology": topology,
                "kappa": kappa,
                "random_seed": seed,
                "split": split,
                "observation_timestep": OBSERVATION_TIMESTEP,
                "large_cascade": outcome["final_activated_proportion"] >= GLOBAL_THRESHOLD,
                **features,
                **outcome,
            })
    return rows, pairing


def matrix(frame: pd.DataFrame, feature_names: tuple[str, ...]) -> np.ndarray:
    return frame.loc[:, feature_names].to_numpy(float)


def train_model(examples: pd.DataFrame):
    excluded = {
        "topology", "random_seed", "split", "observation_timestep", "large_cascade",
        "final_activated_count", "final_activated_proportion", "cascade_duration", "t50",
    }
    feature_names = tuple(column for column in examples.columns if column not in excluded)
    train = examples[examples.split == "train"]
    validation = examples[examples.split == "validation"]
    train_y = train.large_cascade.astype(int).to_numpy()
    validation_y = validation.large_cascade.astype(int).to_numpy()
    candidates = []
    models = {}
    for l2 in L2_CANDIDATES:
        model = fit_logistic_risk_model(
            matrix(train, feature_names), train_y, feature_names, l2=l2
        )
        probabilities = model.predict_proba(matrix(validation, feature_names))
        metrics = classification_metrics(validation_y, probabilities, 0.5)
        candidates.append({"l2": l2, **metrics})
        models[l2] = model
    candidate_frame = pd.DataFrame(candidates).sort_values(
        ["auc", "brier", "l2"], ascending=[False, True, True]
    )
    selected_l2 = float(candidate_frame.iloc[0].l2)
    model = models[selected_l2]
    validation_probabilities = model.predict_proba(matrix(validation, feature_names))
    threshold, _ = select_f2_threshold(validation_y, validation_probabilities)
    model.threshold = threshold
    return model, candidate_frame


def evaluate_classifier(examples: pd.DataFrame, model) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = examples.copy()
    probabilities = model.predict_proba(matrix(predictions, model.feature_names))
    predictions["risk_probability"] = probabilities
    predictions["predicted_large_cascade"] = probabilities >= model.threshold
    rows = []
    for split in ("train", "validation", "test"):
        part = predictions[predictions.split == split]
        rows.append({"split": split, "scope": "all_kappa", **classification_metrics(
            part.large_cascade.astype(int).to_numpy(),
            part.risk_probability.to_numpy(float),
            model.threshold,
        )})
        center = part[part.kappa == INTERVENTION_KAPPA]
        rows.append({"split": split, "scope": "kappa_0.5", **classification_metrics(
            center.large_cascade.astype(int).to_numpy(),
            center.risk_probability.to_numpy(float),
            model.threshold,
        )})
    return predictions, pd.DataFrame(rows)


def run_intervention_seed(seed: int, test_predictions: list[dict]) -> tuple[list[dict], list[dict]]:
    lookup = {(row["topology"], float(row["kappa"])): row for row in test_predictions}
    rows, timesteps = [], []
    for topology in SUPPORTED_TOPOLOGIES:
        simulation = create_simulation(
            topology,
            n=N,
            target_avg_degree=K,
            network_seed=seed,
            agent_seed=seed,
        )
        ignition = simulation.rng.choice(simulation.kol_idx, size=NUM_IGNITE, replace=False)
        paths = build_exposure_paths(simulation)
        prediction = lookup[(topology, INTERVENTION_KAPPA)]
        baseline = run_intervention(
            simulation,
            INTERVENTION_KAPPA,
            InterventionSpec(),
            intervention_seed=seed * 10_000,
            ignition_nodes=ignition,
            prepared_paths=paths,
        )
        baseline_metrics = cascade_metrics(baseline.activated_counts)
        rows.append({
            "topology": topology,
            "random_seed": seed,
            "kappa": INTERVENTION_KAPPA,
            "policy": "none",
            "intervention_budget": 0.0,
            "intervention_weight": 1.0,
            "predicted_risk": bool(prediction["predicted_large_cascade"]),
            "risk_probability": float(prediction["risk_probability"]),
            "baseline_large_cascade": bool(prediction["large_cascade"]),
            "intervention_triggered": False,
            "actual_intervention_time": np.nan,
            "actual_budget_fraction": 0.0,
            "selected_path_count": 0,
            "total_path_count": paths.count,
            **baseline_metrics,
        })

        for budget in BUDGETS:
            for weight in WEIGHTS:
                for policy in POLICIES:
                    should_act = True
                    if policy == "risk_t5":
                        should_act = bool(prediction["predicted_large_cascade"])
                    elif policy == "oracle_t5":
                        should_act = bool(prediction["large_cascade"])

                    if not should_act:
                        result = baseline
                    else:
                        kwargs = {"trigger_progress": 0.05} if policy == "fixed_5pct" else {"trigger_timestep": OBSERVATION_TIMESTEP}
                        spec = InterventionSpec(
                            strategy="observable_momentum",
                            budget=budget,
                            intervention_weight=weight,
                            **kwargs,
                        )
                        result = run_intervention(
                            simulation,
                            INTERVENTION_KAPPA,
                            spec,
                            intervention_seed=seed * 10_000 + 701,
                            ignition_nodes=ignition,
                            prepared_paths=paths,
                            dynamic_selector=select_cascade_aware_paths,
                        )
                    outcome = cascade_metrics(result.activated_counts)
                    common = {
                        "topology": topology,
                        "random_seed": seed,
                        "kappa": INTERVENTION_KAPPA,
                        "policy": policy,
                        "intervention_budget": budget,
                        "intervention_weight": weight,
                        "predicted_risk": bool(prediction["predicted_large_cascade"]),
                        "risk_probability": float(prediction["risk_probability"]),
                        "baseline_large_cascade": bool(prediction["large_cascade"]),
                        "intervention_triggered": bool(should_act and result.intervention_triggered),
                        "actual_intervention_time": result.actual_timestep if should_act else np.nan,
                        "actual_budget_fraction": result.selected_path_count / paths.count if should_act else 0.0,
                        "selected_path_count": result.selected_path_count if should_act else 0,
                        "total_path_count": paths.count,
                    }
                    rows.append({**common, **outcome})
                    for timestep, count in enumerate(result.activated_counts):
                        timesteps.append({
                            **{key: common[key] for key in (
                                "topology", "random_seed", "kappa", "policy",
                                "intervention_budget", "intervention_weight"
                            )},
                            "timestep": timestep,
                            "activated_count": int(count),
                            "activated_proportion": count / N,
                        })
    return rows, timesteps


def summarize_interventions(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = frame[frame.policy == "none"][[
        "topology", "random_seed", "final_activated_proportion"
    ]].rename(columns={"final_activated_proportion": "baseline_final"})
    active = frame[frame.policy != "none"].merge(
        baseline, on=["topology", "random_seed"], validate="many_to_one"
    )
    active["paired_reduction"] = active.baseline_final - active.final_activated_proportion
    keys = ["policy", "intervention_budget", "intervention_weight"]
    rows = []
    topology_rows = []
    for grouping, destination in ((keys, rows), (["topology", *keys], topology_rows)):
        for key, group in active.groupby(grouping, sort=True):
            key = key if isinstance(key, tuple) else (key,)
            reduction = group.paired_reduction.to_numpy(float)
            half = stats.t.ppf(0.975, len(group) - 1) * reduction.std(ddof=1) / np.sqrt(len(group))
            mean_cost = group.actual_budget_fraction.mean()
            destination.append({
                **dict(zip(grouping, key)),
                "n_runs": len(group),
                "intervention_rate": group.intervention_triggered.mean(),
                "mean_actual_budget_fraction": mean_cost,
                "mean_final_cascade_size": group.final_activated_proportion.mean(),
                "global_cascade_probability": (group.final_activated_proportion >= GLOBAL_THRESHOLD).mean(),
                "mean_paired_reduction": reduction.mean(),
                "reduction_ci95_lower": reduction.mean() - half,
                "reduction_ci95_upper": reduction.mean() + half,
                "reduction_per_one_percent_path_budget": reduction.mean() / (100 * mean_cost) if mean_cost else np.nan,
            })
    return pd.DataFrame(rows), pd.DataFrame(topology_rows)


def plot_results(predictions: pd.DataFrame, metrics: pd.DataFrame, summary: pd.DataFrame, output_dir: Path, risk_threshold: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    test = predictions[predictions.split == "test"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for label, color in ((False, "#4C78A8"), (True, "#E45756")):
        values = test.loc[test.large_cascade == label, "risk_probability"]
        axes[0].hist(values, bins=np.linspace(0, 1, 16), alpha=0.65, color=color, label=f"large={label}")
    axes[0].axvline(risk_threshold, color="black", ls="--", label="validation threshold")
    axes[0].set(xlabel="Predicted large-cascade risk", ylabel="Held-out runs", title="Held-out risk-score separation")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    test05 = test[test.kappa == INTERVENTION_KAPPA]
    jitter = np.random.default_rng(42).normal(0, 0.025, len(test05))
    x = test05.topology.map({name: i for i, name in enumerate(SUPPORTED_TOPOLOGIES)}).to_numpy() + jitter
    axes[1].scatter(x, test05.risk_probability, c=np.where(test05.large_cascade, "#E45756", "#4C78A8"), alpha=0.8)
    axes[1].set_xticks(range(len(SUPPORTED_TOPOLOGIES)), SUPPORTED_TOPOLOGIES)
    axes[1].set(xlabel="Topology", ylabel="Predicted risk", title="Held-out κ=.5 predictions")
    axes[1].grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "figure_1_early_warning_heldout.png", dpi=180)
    plt.close(fig)

    focus = summary[summary.intervention_weight == 0.75]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = {"fixed_5pct": "#F58518", "all_t5": "#54A24B", "risk_t5": "#4C78A8", "oracle_t5": "#E45756"}
    for axis, budget in zip(axes, BUDGETS):
        part = focus[focus.intervention_budget == budget]
        for _, row in part.iterrows():
            axis.scatter(100 * row.mean_actual_budget_fraction, row.mean_final_cascade_size, s=85, color=colors[row.policy], label=row.policy)
            axis.annotate(row.policy, (100 * row.mean_actual_budget_fraction, row.mean_final_cascade_size), xytext=(5, 5), textcoords="offset points", fontsize=8)
        axis.set_title(f"Requested budget={budget:.0%}; selected weight=.75")
        axis.set_xlabel("Mean paths acted on across all runs (%)")
        axis.set_ylabel("Mean final activated proportion")
        axis.grid(alpha=0.25)
    fig.suptitle("Held-out cost versus cascade outcome")
    fig.tight_layout()
    fig.savefig(output_dir / "figure_2_risk_trigger_cost_effectiveness.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for axis, weight in zip(axes, WEIGHTS):
        part = summary[summary.intervention_weight == weight]
        x = np.arange(len(POLICIES))
        width = 0.35
        for offset, budget in ((-width / 2, 0.02), (width / 2, 0.05)):
            group = part.set_index(["policy", "intervention_budget"])
            values = [group.loc[(policy, budget), "mean_final_cascade_size"] for policy in POLICIES]
            axis.bar(x + offset, values, width, label=f"budget={budget:.0%}")
        axis.set_xticks(x, POLICIES, rotation=25)
        axis.set_title(f"Selected path weight={weight}")
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Mean final activated proportion")
    axes[0].legend()
    fig.suptitle("Held-out intervention-policy comparison")
    fig.tight_layout()
    fig.savefig(output_dir / "figure_3_policy_comparison.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    all_seeds = (*TRAIN_SEEDS, *VALIDATION_SEEDS, *TEST_SEEDS)
    feature_results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(generate_feature_seed, seed): seed for seed in all_seeds}
        for future in as_completed(futures):
            seed = futures[future]
            feature_results[seed] = future.result()
            print(f"Feature seed {seed} ({len(feature_results)}/{len(all_seeds)})", flush=True)
    feature_rows, pairing_rows = [], []
    for seed in sorted(feature_results):
        rows, pairing = feature_results[seed]
        feature_rows.extend(rows)
        pairing_rows.extend(pairing)
    examples = pd.DataFrame(feature_rows)
    model, model_selection = train_model(examples)
    predictions, classifier_metrics = evaluate_classifier(examples, model)

    test_records = predictions[(predictions.split == "test") & (predictions.kappa == INTERVENTION_KAPPA)].to_dict("records")
    by_seed = {seed: [row for row in test_records if row["random_seed"] == seed] for seed in TEST_SEEDS}
    intervention_results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_intervention_seed, seed, by_seed[seed]): seed for seed in TEST_SEEDS}
        for future in as_completed(futures):
            seed = futures[future]
            intervention_results[seed] = future.result()
            print(f"Intervention test seed {seed} ({len(intervention_results)}/{len(TEST_SEEDS)})", flush=True)
    intervention_rows, timestep_rows = [], []
    for seed in sorted(intervention_results):
        rows, timesteps = intervention_results[seed]
        intervention_rows.extend(rows)
        timestep_rows.extend(timesteps)
    interventions = pd.DataFrame(intervention_rows)
    summary, topology_summary = summarize_interventions(interventions)

    examples.to_csv(args.output_dir / "early_warning_feature_dataset.csv", index=False)
    predictions.to_csv(args.output_dir / "early_warning_predictions.csv", index=False)
    classifier_metrics.to_csv(args.output_dir / "classification_metrics.csv", index=False)
    model_selection.to_csv(args.output_dir / "model_selection.csv", index=False)
    pd.DataFrame(pairing_rows).to_csv(args.output_dir / "paired_randomness_verification.csv", index=False)
    interventions.to_csv(args.output_dir / "risk_intervention_raw_runs.csv", index=False)
    pd.DataFrame(timestep_rows).to_csv(args.output_dir / "risk_intervention_raw_timesteps.csv", index=False)
    summary.to_csv(args.output_dir / "risk_intervention_summary.csv", index=False)
    topology_summary.to_csv(args.output_dir / "risk_intervention_topology_summary.csv", index=False)
    (args.output_dir / "early_warning_model.json").write_text(json.dumps(model.to_dict(), indent=2), encoding="utf-8")
    plot_results(predictions, classifier_metrics, summary, args.output_dir, model.threshold)

    config = {
        "n": N,
        "target_average_degree": K,
        "topologies": list(SUPPORTED_TOPOLOGIES),
        "train_seeds": list(TRAIN_SEEDS),
        "validation_seeds": list(VALIDATION_SEEDS),
        "test_seeds": list(TEST_SEEDS),
        "training_kappas": list(KAPPAS),
        "intervention_kappa": INTERVENTION_KAPPA,
        "observation_timestep": OBSERVATION_TIMESTEP,
        "large_cascade_threshold": GLOBAL_THRESHOLD,
        "budgets": list(BUDGETS),
        "weights": list(WEIGHTS),
        "policies": list(POLICIES),
        "selected_l2": model.l2,
        "selected_risk_threshold": model.threshold,
        "feature_run_count": len(examples),
        "heldout_intervention_run_count": len(interventions),
        "elapsed_seconds": time.time() - started,
        "software_versions": {
            "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "networkx": nx.__version__,
        },
    }
    (args.output_dir / "experiment_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print(json.dumps(config, indent=2), flush=True)


if __name__ == "__main__":
    main()
