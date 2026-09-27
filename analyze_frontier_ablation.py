"""Paired statistics, resource accounting, figures, and report for frontier ablations."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_DIR = Path("outputs") / "frontier_ablation_70seeds"
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260925
METHOD_ORDER = (
    "static_random",
    "static_degree",
    "frontier_random",
    "frontier_degree",
    "observable_pressure",
    "no_momentum",
    "no_pressure",
    "no_degree",
    "no_community",
    "full_momentum",
    "oracle_margin",
)
METHOD_LABELS = {
    "static_random": "Static Random",
    "static_degree": "Static Degree",
    "frontier_random": "Frontier Random",
    "frontier_degree": "Frontier Degree",
    "observable_pressure": "Observable Pressure",
    "no_momentum": "Momentum: no momentum",
    "no_pressure": "Momentum: no pressure",
    "no_degree": "Momentum: no degree",
    "no_community": "Momentum: no community",
    "full_momentum": "Observable Momentum",
    "oracle_margin": "Oracle Margin",
}
COST_COLUMNS = (
    "path_coverage_cost",
    "intensity_adjusted_cost",
    "temporal_suppression_cost",
    "active_source_suppression_cost",
    "frontier_suppression_cost",
)


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    indices = rng.integers(0, values.size, size=(BOOTSTRAP_REPLICATES, values.size))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    records = []
    keys = ["topology", "method"]
    for (topology, method), group in frame.groupby(keys, sort=True):
        final = group.final_activated_proportion.to_numpy(float)
        reduction = group.cascade_reduction.to_numpy(float)
        final_ci = bootstrap_mean_ci(final, rng)
        reduction_ci = bootstrap_mean_ci(reduction, rng)
        record = {
            "topology": topology,
            "method": method,
            "n_runs": len(group),
            "mean_final_cascade_size": final.mean(),
            "median_final_cascade_size": np.median(final),
            "sd_final_cascade_size": final.std(ddof=1),
            "final_ci95_lower": final_ci[0],
            "final_ci95_upper": final_ci[1],
            "large_cascade_probability": group.global_cascade.mean(),
            "mean_paired_reduction": reduction.mean(),
            "reduction_ci95_lower": reduction_ci[0],
            "reduction_ci95_upper": reduction_ci[1],
            "mean_duration": group.cascade_duration.mean(),
            "trigger_rate": group.intervention_triggered.mean(),
        }
        for cost in COST_COLUMNS:
            mean_cost = group[cost].mean()
            record[f"mean_{cost}"] = mean_cost
            record[f"reduction_per_mean_{cost}"] = (
                reduction.mean() / mean_cost if mean_cost > 0 else np.nan
            )
        records.append(record)
    return pd.DataFrame(records)


def paired_contrasts(frame: pd.DataFrame) -> pd.DataFrame:
    wide = frame.pivot(
        index=["topology", "random_seed"],
        columns="method",
        values="final_activated_proportion",
    ).reset_index()
    rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    records = []
    comparators = [method for method in METHOD_ORDER if method != "full_momentum"]
    for topology in (*sorted(frame.topology.unique()), "ALL"):
        part = wide if topology == "ALL" else wide[wide.topology == topology]
        for comparator in comparators:
            raw = part.assign(
                difference=part.full_momentum - part[comparator]
            )[["random_seed", "difference"]]
            if topology == "ALL":
                # Resample seed clusters; each sampled seed contributes all topologies.
                by_seed = raw.groupby("random_seed").difference.mean().sort_index()
                values = by_seed.to_numpy(float)
            else:
                values = raw.difference.to_numpy(float)
            ci = bootstrap_mean_ci(values, rng)
            sd = values.std(ddof=1)
            records.append(
                {
                    "topology": topology,
                    "comparison": f"full_momentum - {comparator}",
                    "comparator": comparator,
                    "n_seed_clusters": values.size,
                    "mean_paired_difference": values.mean(),
                    "median_paired_difference": np.median(values),
                    "sd_paired_difference": sd,
                    "paired_effect_size_dz": values.mean() / sd if sd > 0 else np.nan,
                    "bootstrap_ci95_lower": ci[0],
                    "bootstrap_ci95_upper": ci[1],
                    "momentum_better_fraction": (values < 0).mean(),
                    "ties_fraction": (values == 0).mean(),
                }
            )
    return pd.DataFrame(records)


def planned_mechanism_contrasts(frame: pd.DataFrame) -> pd.DataFrame:
    """Contrasts that identify candidate filtering and each scoring increment."""
    wide = frame.pivot(
        index=["topology", "random_seed"],
        columns="method",
        values="final_activated_proportion",
    ).reset_index()
    plans = (
        ("frontier_random", "static_random", "frontier filtering vs static random"),
        ("frontier_random", "static_degree", "frontier filtering vs static degree"),
        ("frontier_degree", "frontier_random", "degree ranking within frontier"),
        ("observable_pressure", "frontier_degree", "pressure score beyond frontier degree"),
        ("full_momentum", "observable_pressure", "full score beyond pressure score"),
        ("full_momentum", "no_momentum", "incremental momentum feature"),
        ("full_momentum", "no_pressure", "incremental pressure feature"),
        ("full_momentum", "no_degree", "incremental degree features"),
        ("full_momentum", "no_community", "incremental community feature"),
        ("full_momentum", "oracle_margin", "observable-to-oracle gap"),
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED + 2)
    records = []
    for topology in (*sorted(frame.topology.unique()), "ALL"):
        part = wide if topology == "ALL" else wide[wide.topology == topology]
        for left, right, estimand in plans:
            raw = part.assign(difference=part[left] - part[right])[["random_seed", "difference"]]
            values = (
                raw.groupby("random_seed").difference.mean().sort_index().to_numpy(float)
                if topology == "ALL"
                else raw.difference.to_numpy(float)
            )
            ci = bootstrap_mean_ci(values, rng)
            sd = values.std(ddof=1)
            records.append({
                "topology": topology,
                "estimand": estimand,
                "left_method": left,
                "right_method": right,
                "difference_definition": f"{left} - {right}",
                "n_seed_clusters": values.size,
                "mean_paired_difference": values.mean(),
                "median_paired_difference": np.median(values),
                "sd_paired_difference": sd,
                "paired_effect_size_dz": values.mean() / sd if sd > 0 else np.nan,
                "bootstrap_ci95_lower": ci[0],
                "bootstrap_ci95_upper": ci[1],
                "left_better_fraction": (values < 0).mean(),
                "ties_fraction": (values == 0).mean(),
            })
    return pd.DataFrame(records)


def cost_contrasts(frame: pd.DataFrame) -> pd.DataFrame:
    index = ["topology", "random_seed"]
    records = []
    for cost in COST_COLUMNS:
        wide = frame.pivot(index=index, columns="method", values=cost).reset_index()
        for comparator in ("static_degree", "frontier_random", "frontier_degree", "observable_pressure"):
            values = (wide.full_momentum - wide[comparator]).to_numpy(float)
            ci = bootstrap_mean_ci(values, np.random.default_rng(BOOTSTRAP_SEED + len(records) + 100))
            records.append(
                {
                    "cost_metric": cost,
                    "comparison": f"full_momentum - {comparator}",
                    "mean_paired_cost_difference": values.mean(),
                    "bootstrap_ci95_lower": ci[0],
                    "bootstrap_ci95_upper": ci[1],
                }
            )
    return pd.DataFrame(records)


def plots(summary: pd.DataFrame, contrasts: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = plt.cm.tab20(np.linspace(0, 1, len(METHOD_ORDER)))
    color_map = dict(zip(METHOD_ORDER, colors))
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), sharey=True)
    for axis, topology in zip(axes.ravel(), ("ER", "WS", "BA", "SBM")):
        part = summary[summary.topology == topology].set_index("method").reindex(METHOD_ORDER).dropna(how="all")
        x = np.arange(len(part))
        means = part.mean_final_cascade_size.to_numpy(float)
        low = means - part.final_ci95_lower.to_numpy(float)
        high = part.final_ci95_upper.to_numpy(float) - means
        axis.bar(x, means, color=[color_map[m] for m in part.index])
        axis.errorbar(x, means, yerr=np.vstack([low, high]), fmt="none", color="black", capsize=2)
        axis.set_xticks(x, [METHOD_LABELS[m] for m in part.index], rotation=55, ha="right")
        axis.set_title(topology)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Fair frontier baselines and Momentum ablations\nκ=.5, trigger=5%, budget=5%, selected-path weight=.75")
    fig.supylabel("Mean final activated proportion (95% seed-bootstrap CI)")
    fig.tight_layout()
    fig.savefig(output_dir / "figure_1_method_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    focus = contrasts[
        (contrasts.topology == "ALL")
        & contrasts.comparator.isin([
            "static_degree", "frontier_random", "frontier_degree",
            "observable_pressure", "no_momentum", "no_pressure",
            "no_degree", "no_community", "oracle_margin",
        ])
    ].copy()
    focus["label"] = focus.comparator.map(METHOD_LABELS)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    panels = (
        (axes[0], focus, "All planned comparators"),
        (axes[1], focus[focus.comparator != "static_degree"], "Zoom: frontier and feature comparisons"),
    )
    for axis, part, title in panels:
        y = np.arange(len(part))
        means = part.mean_paired_difference.to_numpy(float)
        low = means - part.bootstrap_ci95_lower.to_numpy(float)
        high = part.bootstrap_ci95_upper.to_numpy(float) - means
        axis.errorbar(means, y, xerr=np.vstack([low, high]), fmt="o", capsize=4, color="#4C78A8")
        axis.axvline(0, color="black", linewidth=1)
        axis.set_yticks(y, part.label)
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.25)
    fig.suptitle("Paired seed-cluster contrasts across four topologies")
    fig.supxlabel("Full Momentum − comparator final cascade (negative favors Momentum)")
    fig.tight_layout()
    fig.savefig(output_dir / "figure_2_paired_differences.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    primary_methods = (
        "static_random", "static_degree", "frontier_random", "frontier_degree",
        "observable_pressure", "full_momentum", "oracle_margin",
    )
    pooled = summary[summary.method.isin(primary_methods)].groupby("method", as_index=False).agg(
        mean_final=("mean_final_cascade_size", "mean"),
        intensity_cost=("mean_intensity_adjusted_cost", "mean"),
        temporal_cost=("mean_temporal_suppression_cost", "mean"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), sharey=True)
    for axis, cost, label in (
        (axes[0], "intensity_cost", "Mean intensity-adjusted path cost"),
        (axes[1], "temporal_cost", "Mean temporal suppression cost"),
    ):
        for row in pooled.itertuples(index=False):
            x = getattr(row, cost)
            axis.scatter(x, row.mean_final, s=65, color=color_map[row.method], label=METHOD_LABELS[row.method])
        axis.set_xlabel(label)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Mean final activated proportion")
    fig.suptitle("Cost–outcome comparison (lower-left is preferred within each panel)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(output_dir / "figure_3_cost_outcome.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, decimals: int = 4) -> str:
    formatted = frame.copy()
    for column in formatted.select_dtypes(include=["float"]).columns:
        formatted[column] = formatted[column].map(lambda value: f"{value:.{decimals}f}")
    lines = [
        "| " + " | ".join(formatted.columns) + " |",
        "|" + "|".join(["---"] * len(formatted.columns)) + "|",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in formatted.itertuples(index=False, name=None))
    return "\n".join(lines)


def write_report(
    summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    planned: pd.DataFrame,
    output_dir: Path,
) -> None:
    primary = summary[[
        "topology", "method", "mean_final_cascade_size", "large_cascade_probability",
        "mean_paired_reduction", "mean_path_coverage_cost",
        "mean_intensity_adjusted_cost", "mean_temporal_suppression_cost",
    ]]
    primary = primary[primary.method.isin([
        "static_degree", "frontier_random", "frontier_degree", "observable_pressure",
        "full_momentum", "oracle_margin",
    ])]
    contrast_focus = contrasts[
        (contrasts.topology == "ALL")
        & contrasts.comparator.isin([
            "frontier_random", "frontier_degree", "observable_pressure",
            "no_momentum", "no_pressure", "no_degree", "no_community", "oracle_margin",
        ])
    ][[
        "comparison", "mean_paired_difference", "bootstrap_ci95_lower",
        "bootstrap_ci95_upper", "paired_effect_size_dz", "momentum_better_fraction",
    ]]
    planned_focus = planned[planned.topology == "ALL"][[
        "estimand", "difference_definition", "mean_paired_difference",
        "bootstrap_ci95_lower", "bootstrap_ci95_upper", "paired_effect_size_dz",
        "left_better_fraction",
    ]]
    report = f"""# Frontier-matched Baselines and Observable Momentum Ablation

## 研究目的

本實驗檢查 Observable Momentum 的優勢究竟來自 live-frontier filtering，還是 Momentum 評分本身。所有方法固定 κ=.5、5% progress trigger、5% path budget、selected-path weight=.75、相同 topology／seed／agent／threshold／KOL／ignition。

## 方法定義

- Static Degree：從全網路依兩端 degree 選邊。
- Frontier Random：只在 active-to-inactive paths 中隨機選擇。
- Frontier Degree：只在相同 live frontier 中依兩端 degree 排序。
- Observable Pressure：frontier local pressure 加 degree／community。
- Observable Momentum：Pressure 加上 sender 是否剛 activation。
- Ablations：分別移除 momentum、pressure、degree、community，移除後將保留係數依比例正規化。
- Oracle Margin：使用真實 latent threshold 的 privileged benchmark，並非全域最優保證。

## 成本定義

- path coverage = selected paths / all directed exposure paths。
- intensity-adjusted cost = path coverage × (1 − selected path weight)。
- temporal suppression cost = 每個受影響 transition 的 suppression path-weight 總量 / all paths。
- active-source suppression cost：只累計當時 sender 已 active 的被抑制路徑。
- frontier suppression cost：只累計當時仍為 active-to-inactive 的被抑制路徑。

前兩項是平台配置資源；temporal cost 是維持介入的 path-step 負擔。後兩項較適合解讀為實際送達 active sources／live frontier 的 suppression volume，不應單獨視為平台支出。

## Topology-level summary

{markdown_table(primary)}

## Paired Momentum contrasts

Difference 定義為 Full Momentum − comparator；負值代表 Full Momentum final cascade 較低。CI 以 seed cluster bootstrap 10,000 次計算，ALL 先在同 seed 內平均四種 topology，再重抽 seed。

{markdown_table(contrast_focus)}

## Planned mechanism contrasts

每列差值定義為 left − right；負值代表 left 方法的 final cascade 較低。這些 contrasts 依照「候選集合 → frontier 內結構排序 → pressure → momentum feature」的順序識別效果來源。

{markdown_table(planned_focus)}

## 實際研究結論

- 最大改善來自 **live-frontier candidate filtering**。Frontier Random 相較兩個 static controls 大幅降低 final cascade，表示原先 dynamic-static 差距主要不能歸因於 Momentum scoring。
- Full Momentum 相對 Frontier Random／Degree 仍有小幅平均改善，但量級只有約 0.0016–0.0017 final proportion，遠小於 frontier filtering 本身。
- Full Momentum 相對 no-momentum 的 CI 包含 0；相對 Observable Pressure 的 CI 也包含 0。因此目前沒有穩定證據證明「newly-active sender」特徵提供獨立增益。
- Degree features 的 pooled contrast很小但 CI 排除 0；其方向在個別 topology 並不完全一致，應視為次要且具結構依賴的效果。
- Community feature 幾乎沒有可辨識的增益；在非 SBM topology 本來即為 0，SBM contrast 也跨過 0。
- Oracle 的 final cascade 平均仍比 Full Momentum 低約 0.0021，顯示 latent threshold information 有小幅額外價值。
- 因此現階段應把主要方法稱為 **cascade-aware frontier intervention**。Observable Momentum 可保留為其中一個 scoring variant，但不應宣稱 momentum feature 是已被驗證的核心創新。

## 解讀原則

- 若 Momentum 優於 Static Degree 但不優於 Frontier Random／Degree，主要成果應歸因於 frontier filtering。
- 若 Full Momentum 優於 Frontier baselines，但不優於 no-momentum，則額外效果來自 pressure／degree，而非 momentum feature。
- 只有當 Full Momentum 相對 Frontier baselines 與 no-momentum 的 paired CI 均排除 0，才能主張 momentum feature 有穩定額外價值。
- Oracle 僅提供 privileged-information benchmark；Full Momentum 不必優於 Oracle。
"""
    (output_dir / "frontier_ablation_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    frame = pd.read_csv(args.output_dir / "raw_run_results.csv")
    expected = len(frame.topology.unique()) * len(frame.random_seed.unique()) * len(METHOD_ORDER)
    if len(frame) != expected:
        raise ValueError(f"Expected {expected} complete rows, found {len(frame)}")
    if frame.duplicated(["topology", "random_seed", "method"]).any():
        raise ValueError("Duplicate topology-seed-method rows")
    summary = summarize(frame)
    contrasts = paired_contrasts(frame)
    costs = cost_contrasts(frame)
    planned = planned_mechanism_contrasts(frame)
    summary.to_csv(args.output_dir / "method_summary.csv", index=False)
    contrasts.to_csv(args.output_dir / "paired_method_contrasts.csv", index=False)
    costs.to_csv(args.output_dir / "paired_cost_contrasts.csv", index=False)
    planned.to_csv(args.output_dir / "planned_mechanism_contrasts.csv", index=False)
    plots(summary, contrasts, args.output_dir)
    write_report(summary, contrasts, planned, args.output_dir)
    print(f"Analyzed {len(frame)} runs", flush=True)


if __name__ == "__main__":
    main()
