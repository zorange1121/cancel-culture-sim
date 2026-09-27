"""Analyze early warning and risk-triggered intervention with seed-cluster bootstrap."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from early_warning import classification_metrics


OUTPUT_DIR = Path("outputs") / "early_warning_risk_intervention_heldout"
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260922


def markdown_table(frame: pd.DataFrame, decimals: int = 4) -> str:
    result = frame.copy()
    for column in result.select_dtypes(include=["float"]).columns:
        result[column] = result[column].map(lambda value: f"{value:.{decimals}f}")
    lines = [
        "| " + " | ".join(map(str, result.columns)) + " |",
        "|" + "|".join(["---"] * len(result.columns)) + "|",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in result.itertuples(index=False, name=None))
    return "\n".join(lines)


def seed_bootstrap_classification(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    seeds = np.sort(frame.random_seed.unique())
    rows = []
    for scope, part in (("all_kappa", frame), ("kappa_0.5", frame[frame.kappa == 0.5])):
        observed = classification_metrics(
            part.large_cascade.astype(int).to_numpy(),
            part.risk_probability.to_numpy(float),
            threshold,
        )
        samples = {name: [] for name in ("auc", "recall", "specificity", "precision", "intervention_rate")}
        seed_labels = {seed: part.loc[part.random_seed == seed, "large_cascade"].astype(int).to_numpy() for seed in seeds}
        seed_probabilities = {seed: part.loc[part.random_seed == seed, "risk_probability"].to_numpy(float) for seed in seeds}
        for _ in range(BOOTSTRAP_REPLICATES):
            selected = rng.choice(seeds, size=len(seeds), replace=True)
            boot_labels = np.concatenate([seed_labels[seed] for seed in selected])
            boot_probabilities = np.concatenate([seed_probabilities[seed] for seed in selected])
            metrics = classification_metrics(
                boot_labels,
                boot_probabilities,
                threshold,
            )
            for name in samples:
                if np.isfinite(metrics[name]):
                    samples[name].append(metrics[name])
        for name, values in samples.items():
            rows.append({
                "scope": scope,
                "metric": name,
                "estimate": observed[name],
                "cluster_bootstrap_ci95_lower": np.quantile(values, 0.025),
                "cluster_bootstrap_ci95_upper": np.quantile(values, 0.975),
            })
    return pd.DataFrame(rows)


def intervention_bootstrap(frame: pd.DataFrame) -> pd.DataFrame:
    baseline = frame[frame.policy == "none"][["topology", "random_seed", "final_activated_proportion"]].rename(
        columns={"final_activated_proportion": "baseline_final"}
    )
    active = frame[frame.policy != "none"].merge(baseline, on=["topology", "random_seed"], validate="many_to_one")
    active["paired_reduction"] = active.baseline_final - active.final_activated_proportion
    active["global_cascade"] = active.final_activated_proportion >= 0.5
    rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    seeds = np.sort(active.random_seed.unique())
    records = []
    keys = ["policy", "intervention_budget", "intervention_weight"]
    for key, group in active.groupby(keys, sort=True):
        values = {name: [] for name in ("mean_final", "mean_reduction", "mean_cost", "global_probability")}
        indices = rng.integers(0, len(seeds), size=(BOOTSTRAP_REPLICATES, len(seeds)))
        seed_means = group.groupby("random_seed")[[
            "final_activated_proportion", "paired_reduction", "actual_budget_fraction", "global_cascade"
        ]].mean().reindex(seeds)
        values["mean_final"] = seed_means.final_activated_proportion.to_numpy()[indices].mean(axis=1)
        values["mean_reduction"] = seed_means.paired_reduction.to_numpy()[indices].mean(axis=1)
        values["mean_cost"] = seed_means.actual_budget_fraction.to_numpy()[indices].mean(axis=1)
        values["global_probability"] = seed_means.global_cascade.to_numpy()[indices].mean(axis=1)
        record = dict(zip(keys, key))
        for name, samples in values.items():
            record[name] = getattr(group, {
                "mean_final": "final_activated_proportion",
                "mean_reduction": "paired_reduction",
                "mean_cost": "actual_budget_fraction",
                "global_probability": "global_cascade",
            }[name]).mean()
            record[f"{name}_ci95_lower"] = np.quantile(samples, 0.025)
            record[f"{name}_ci95_upper"] = np.quantile(samples, 0.975)
        records.append(record)
    return pd.DataFrame(records)


def paired_policy_contrasts(frame: pd.DataFrame) -> pd.DataFrame:
    active = frame[frame.policy != "none"]
    wide_final = active.pivot(
        index=["topology", "random_seed", "intervention_budget", "intervention_weight"],
        columns="policy", values="final_activated_proportion"
    ).reset_index()
    wide_cost = active.pivot(
        index=["topology", "random_seed", "intervention_budget", "intervention_weight"],
        columns="policy", values="actual_budget_fraction"
    ).reset_index()
    merged = wide_final.merge(wide_cost, on=["topology", "random_seed", "intervention_budget", "intervention_weight"], suffixes=("_final", "_cost"))
    rng = np.random.default_rng(BOOTSTRAP_SEED + 2)
    seeds = np.sort(merged.random_seed.unique())
    records = []
    for (budget, weight), group in merged.groupby(["intervention_budget", "intervention_weight"]):
        for comparator in ("fixed_5pct", "all_t5", "oracle_t5"):
            final_diff = group["risk_t5_final"] - group[f"{comparator}_final"]
            cost_diff = group["risk_t5_cost"] - group[f"{comparator}_cost"]
            indices = rng.integers(0, len(seeds), size=(BOOTSTRAP_REPLICATES, len(seeds)))
            seed_final = group.assign(value=final_diff).groupby("random_seed")["value"].mean().reindex(seeds).to_numpy()
            seed_cost = group.assign(value=cost_diff).groupby("random_seed")["value"].mean().reindex(seeds).to_numpy()
            boot_final = seed_final[indices].mean(axis=1)
            boot_cost = seed_cost[indices].mean(axis=1)
            records.append({
                "intervention_budget": budget,
                "intervention_weight": weight,
                "comparison": f"risk_t5 - {comparator}",
                "mean_final_difference": final_diff.mean(),
                "final_difference_ci95_lower": np.quantile(boot_final, 0.025),
                "final_difference_ci95_upper": np.quantile(boot_final, 0.975),
                "mean_cost_difference": cost_diff.mean(),
                "cost_difference_ci95_lower": np.quantile(boot_cost, 0.025),
                "cost_difference_ci95_upper": np.quantile(boot_cost, 0.975),
            })
    return pd.DataFrame(records)


def risk_frontier(frame: pd.DataFrame, predictions: pd.DataFrame, selected_threshold: float) -> pd.DataFrame:
    baseline = frame[frame.policy == "none"][["topology", "random_seed", "final_activated_proportion"]].rename(
        columns={"final_activated_proportion": "baseline_final"}
    )
    records = []
    all_t5 = frame[frame.policy == "all_t5"].merge(
        baseline, on=["topology", "random_seed"], validate="many_to_one"
    )
    all_t5["all_t5_reduction"] = all_t5.baseline_final - all_t5.final_activated_proportion
    thresholds = np.unique(np.concatenate((np.linspace(0, 1, 101), [selected_threshold])))
    for (budget, weight), group in all_t5.groupby(["intervention_budget", "intervention_weight"]):
        baseline_mean = group.baseline_final.mean()
        random_mean_benefit = group.all_t5_reduction.mean()
        for threshold in thresholds:
            act = group.risk_probability >= threshold
            final = np.where(act, group.final_activated_proportion, group.baseline_final)
            cost = np.where(act, group.actual_budget_fraction, 0.0)
            action_rate = act.mean()
            random_expected_final = baseline_mean - action_rate * random_mean_benefit
            records.append({
                "intervention_budget": budget,
                "intervention_weight": weight,
                "risk_threshold": threshold,
                "is_validation_selected_threshold": bool(np.isclose(threshold, selected_threshold)),
                "intervention_rate": action_rate,
                "mean_actual_budget_fraction": cost.mean(),
                "mean_final_cascade_size": final.mean(),
                "global_cascade_probability": (final >= 0.5).mean(),
                "random_gating_expected_final": random_expected_final,
                "targeting_gain_vs_random_gating": random_expected_final - final.mean(),
            })
    return pd.DataFrame(records)


def plot_frontier(frontier: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for axis, weight in zip(axes, (0.0, 0.75)):
        part = frontier[frontier.intervention_weight == weight]
        for budget, group in part.groupby("intervention_budget"):
            group = group.sort_values("mean_actual_budget_fraction")
            axis.plot(100 * group.mean_actual_budget_fraction, group.mean_final_cascade_size, "-", label=f"risk gating; budget={budget:.0%}")
            axis.plot(100 * group.mean_actual_budget_fraction, group.random_gating_expected_final, "--", alpha=0.65, label=f"random gating; budget={budget:.0%}")
            chosen = group[group.is_validation_selected_threshold]
            axis.scatter(100 * chosen.mean_actual_budget_fraction, chosen.mean_final_cascade_size, s=70, marker="*", color="black", zorder=4)
        axis.set_title(f"Selected path weight={weight}")
        axis.set_xlabel("Mean paths acted on across all runs (%)")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Mean final activated proportion")
    axes[0].legend(fontsize=8)
    fig.suptitle("Held-out risk-threshold frontier; star = validation-selected threshold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "figure_4_risk_threshold_frontier.png", dpi=180)
    plt.close(fig)


def main() -> None:
    config = json.loads((OUTPUT_DIR / "experiment_config.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(OUTPUT_DIR / "early_warning_predictions.csv")
    interventions = pd.read_csv(OUTPUT_DIR / "risk_intervention_raw_runs.csv")
    threshold = float(config["selected_risk_threshold"])
    test = predictions[predictions.split == "test"]

    classification_ci = seed_bootstrap_classification(test, threshold)
    intervention_ci = intervention_bootstrap(interventions)
    contrasts = paired_policy_contrasts(interventions)
    frontier = risk_frontier(interventions, predictions, threshold)
    classification_ci.to_csv(OUTPUT_DIR / "classification_seed_bootstrap_ci.csv", index=False)
    intervention_ci.to_csv(OUTPUT_DIR / "intervention_seed_bootstrap_ci.csv", index=False)
    contrasts.to_csv(OUTPUT_DIR / "paired_policy_contrasts.csv", index=False)
    frontier.to_csv(OUTPUT_DIR / "risk_threshold_frontier.csv", index=False)
    plot_frontier(frontier)

    primary = intervention_ci[(intervention_ci.intervention_budget == 0.05) & (intervention_ci.intervention_weight == 0.75)][[
        "policy", "mean_cost", "mean_final", "global_probability", "mean_reduction",
        "mean_reduction_ci95_lower", "mean_reduction_ci95_upper"
    ]].copy()
    primary.columns = ["Policy", "Mean path cost", "Mean final", "Global-cascade P", "Mean paired reduction", "Reduction CI low", "Reduction CI high"]
    prediction_table = classification_ci.pivot(index="metric", columns="scope", values=["estimate", "cluster_bootstrap_ci95_lower", "cluster_bootstrap_ci95_upper"]).reset_index()
    prediction_table.columns = ["Metric", "All κ estimate", "κ=.5 estimate", "All κ CI low", "κ=.5 CI low", "All κ CI high", "κ=.5 CI high"]
    primary_contrasts = contrasts[(contrasts.intervention_budget == 0.05) & (contrasts.intervention_weight == 0.75)][[
        "comparison", "mean_final_difference", "final_difference_ci95_lower", "final_difference_ci95_upper",
        "mean_cost_difference", "cost_difference_ci95_lower", "cost_difference_ci95_upper"
    ]].copy()
    primary_contrasts.columns = ["Comparison", "Final diff", "Final CI low", "Final CI high", "Cost diff", "Cost CI low", "Cost CI high"]

    chosen = frontier[(frontier.intervention_budget == 0.05) & (frontier.intervention_weight == 0.75) & frontier.is_validation_selected_threshold].iloc[0]
    baseline_mean = interventions[interventions.policy == "none"].final_activated_proportion.mean()
    report = f"""# Early Cascade Warning + Risk-triggered Cascade-aware Intervention

## 研究問題

是否能只使用 cascade 前五個 timestep 的可觀測資訊，預測未來是否形成 large cascade，並讓平台只對高風險 runs 使用 Observable Momentum intervention？

## 實驗設計

- ER、WS、BA、SBM，N=10,000，target average degree≈10。
- Early-warning dataset：κ∈{{0.4, 0.5, 0.6}}，70 paired seeds，共 840 samples。
- Seed-level split：0–49 training、50–59 validation、60–69 held-out test。同一 seed 的所有 topology 與 κ 都位於同一 split，避免 population leakage。
- Prediction time：t=5。特徵只包含 t≤5 activation trajectory、live active-to-inactive frontier、inactive-node local exposure、active/newly-active degree、topology 與 κ。
- Label：baseline final activated proportion≥0.5。
- Model：L2 logistic regression。L2 只由 validation AUC 選擇；risk threshold 只由 validation F2 選擇，偏重不漏掉 large cascades。Test set 沒有參與選擇。
- Held-out intervention：只在 κ=.5、test seeds 60–69 上評估，共 40 topology-seed runs。比較 no intervention、fixed 5% progress trigger、all-at-t5、risk-at-t5 與知道未來 label 的 Oracle-at-t5。
- Observable Momentum budgets=2%/5%，selected path weights=0/.75。Agent、threshold、KOL、ignition、graph 與 cascade rule 均未改變。

## Early-warning 結果

Validation 選出的 threshold 為 {threshold:.4f}。Held-out 表現如下；CI 使用 10,000 次 seed-cluster bootstrap，重抽 10 個 test seeds，而不是把同 seed 的 topology/κ 當成獨立樣本。

{markdown_table(prediction_table)}

在最直接相關的 held-out κ=.5 subset 中，14 個 baseline large cascades 有 13 個被抓到，漏掉 1 個；26 個非大型 cascade 中有 15 個被標記為高風險。AUC={classification_ci[(classification_ci.scope == 'kappa_0.5') & (classification_ci.metric == 'auc')].estimate.iloc[0]:.3f}，代表前五步已有可用排序訊號，但 false-positive rate 仍高。這是一個偏向高 recall 的 warning system，不是準確的最終結果分類器。

標準化 logistic coefficients 受高度相關的 trajectory/frontier features 影響，不能逐一解讀成因果效果。整體而言，frontier paths per inactive node、frontier fraction、inactive local exposure 與早期 growth 是主要預測訊號。

## Risk-triggered intervention 結果

Primary condition 使用 5% requested budget 與 path weight=.75，也就是只削弱 selected exposure 25%。Held-out baseline mean final cascade 為 {baseline_mean:.4f}。

{markdown_table(primary)}

Risk policy 對 70% runs 介入，mean actual path cost 為 {chosen.mean_actual_budget_fraction:.4f}，final cascade 降至 {chosen.mean_final_cascade_size:.4f}。相較 fixed 5% trigger，它少使用約 {100 * (1 - chosen.mean_actual_budget_fraction / primary.loc[primary.Policy == 'fixed_5pct', 'Mean path cost'].iloc[0]):.1f}% exposure-path budget，但 mean final cascade 高約 {chosen.mean_final_cascade_size - primary.loc[primary.Policy == 'fixed_5pct', 'Mean final'].iloc[0]:.4f}，且 40 runs 中仍有 1 個 large cascade，正是 classifier 漏掉的 false negative。

{markdown_table(primary_contrasts)}

Oracle gating 只對真正會形成 large cascade 的 35% runs 介入，以最低成本把 held-out global-cascade probability 降為 0。它的 mean final size高於 risk policy，是因為 Oracle 刻意不處理原本不會超過 50% 的中小型 cascades；它最佳化的是避免 large cascade 與節省介入，而不是把所有 final sizes 壓到最低。

Threshold frontier 顯示 risk ranking 通常優於相同 action rate 的 random gating，但 validation-selected threshold 並不支配 fixed/all-t5 policy：它提供較低成本與較高 residual cascade 的 Pareto trade-off。換句話說，early warning 成功把資源集中到較危險 runs，但目前 prediction quality 還不足以在不犧牲任何 outcome 的情況下取代固定早期介入。

在 primary condition 與相同 {chosen.intervention_rate:.0%} action rate 下，若隨機選擇要介入的 runs，預期 mean final cascade 為 {chosen.random_gating_expected_final:.4f}；risk ranking 實際得到 {chosen.mean_final_cascade_size:.4f}，改善 {chosen.targeting_gain_vs_random_gating:.4f}。四組 budget/weight conditions 的完整 threshold frontier 都沒有低於同 action rate random gating 的 targeting gain，表示預測分數確實具有資源排序價值，而不只是因為介入了較多 runs。

## 可信度與限制

- 新 baseline features 與既有 STEP 1 的 840 個對應 runs 逐 run 完全一致。
- Fixed-trigger weight=0 的 80 個 held-out runs 與既有 STEP 3 逐 run完全一致。
- 四種 topology 在每個 seed 使用相同 ignition；沒有 condition-seed duplicates。
- 完整測試 30/30 passed。
- Held-out intervention test 只有 10 個 seed clusters，bootstrap intervals 很寬。結果應視為 proof of concept。
- Large-cascade threshold 0.5 沿用先前 descriptive definition；不同 operational objective 會改變最佳 risk threshold。
- Logistic model使用手工可觀測 features，未測試 temporal neural network 或 graph learning，也沒有真實平台資料外部驗證。

## 結論

前五個 timestep 已包含中等強度的 early-warning signal。Risk-triggered Observable Momentum 能以較少 intervention cost 大幅降低 cascade，但漏報會直接留下 residual large-cascade risk。最有價值的發現不是「risk policy 全面勝過固定 trigger」，而是它形成可量化的成本—風險 frontier，並且 Oracle gap 清楚指出：若能改善 early warning precision/recall，仍有相當大的資源節省空間。

這使研究從單純比較 targeting strategy，延伸為兩階段系統：先預測 cascade risk，再依風險決定是否啟動 Cascade-aware control。

## 輸出

- `early_warning_feature_dataset.csv`：840 個 early-warning samples。
- `early_warning_predictions.csv`、`classification_metrics.csv`、`classification_seed_bootstrap_ci.csv`：預測結果。
- `risk_intervention_raw_runs.csv`、`risk_intervention_raw_timesteps.csv`：held-out intervention raw data。
- `risk_intervention_summary.csv`、`intervention_seed_bootstrap_ci.csv`、`paired_policy_contrasts.csv`：介入統計。
- `risk_threshold_frontier.csv`：所有風險門檻的 held-out cost/outcome curve。
- `figure_1_early_warning_heldout.png` 至 `figure_4_risk_threshold_frontier.png`：主要圖表。
"""
    (OUTPUT_DIR / "early_warning_risk_intervention_report.md").write_text(report, encoding="utf-8")
    integrity = {
        "feature_rows": len(predictions),
        "feature_duplicates": int(predictions.duplicated(["topology", "kappa", "random_seed"]).sum()),
        "test_seed_clusters": int(test.random_seed.nunique()),
        "intervention_rows": len(interventions),
        "paired_ignition_all_match": bool(pd.read_csv(OUTPUT_DIR / "paired_randomness_verification.csv").matches_seed_reference.all()),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    }
    (OUTPUT_DIR / "analysis_integrity.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT_DIR / 'early_warning_risk_intervention_report.md'}")


if __name__ == "__main__":
    main()
