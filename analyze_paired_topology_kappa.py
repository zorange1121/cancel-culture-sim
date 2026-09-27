"""Paired-seed statistical analysis of the existing topology x kappa sweep."""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.anova import AnovaRM


INPUT_PATH = (
    Path("outputs")
    / "step1_kappa_sweep_20seeds_corrected_parsing"
    / "raw_run_results.csv"
)
OUTPUT_DIR = Path("outputs") / "step1_paired_topology_kappa_analysis"
TOPOLOGIES = ("ER", "WS", "BA", "SBM")
CONTRASTS = (("WS", "ER"), ("BA", "ER"), ("SBM", "ER"))
FOCUS_TRANSITIONS = ((0.4, 0.5), (0.5, 0.6), (0.6, 0.7))
COLORS = {"ER": "#1f77b4", "WS": "#ff7f0e", "BA": "#2ca02c", "SBM": "#d62728"}


def paired_summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    margin = float(stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n))
    return {
        "n_pairs": n,
        "mean_difference": mean,
        "median_difference": float(np.median(values)),
        "std_difference": sd,
        "ci95_lower": mean - margin,
        "ci95_upper": mean + margin,
        "cohens_dz": mean / sd if sd else math.nan,
        "fraction_positive": float(np.mean(values > 0)),
        "fraction_negative": float(np.mean(values < 0)),
        "fraction_zero": float(np.mean(values == 0)),
        "ci_method": "paired t interval",
    }


def validate_balanced_design(data: pd.DataFrame) -> None:
    required = {
        "topology",
        "kappa",
        "random_seed",
        "final_activated_proportion",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if data.duplicated(["topology", "kappa", "random_seed"]).any():
        raise ValueError("Duplicate topology/kappa/seed cells")
    cell_sizes = data.groupby(["topology", "kappa"]).size()
    if set(cell_sizes) != {20}:
        raise ValueError("Every topology/kappa condition must contain exactly 20 seeds")
    per_seed = data.groupby("random_seed").size()
    if set(per_seed) != {44}:
        raise ValueError("Every seed must contain all 44 topology/kappa cells")


def topology_differences(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    wide = data.pivot(
        index=["random_seed", "kappa"],
        columns="topology",
        values="final_activated_proportion",
    ).reset_index()
    raw_rows = []
    summary_rows = []
    for left, right in CONTRASTS:
        contrast = f"{left} - {right}"
        for kappa, group in wide.groupby("kappa", sort=True):
            differences = (group[left] - group[right]).to_numpy(float)
            for seed, value in zip(group.random_seed, differences):
                raw_rows.append(
                    {
                        "contrast": contrast,
                        "kappa": float(kappa),
                        "random_seed": int(seed),
                        "paired_difference": float(value),
                    }
                )
            summary_rows.append(
                {
                    "contrast": contrast,
                    "kappa": float(kappa),
                    **paired_summary(differences),
                }
            )
    return pd.DataFrame(raw_rows), pd.DataFrame(summary_rows)


def seed_vs_topology_variation(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for kappa, group in data.groupby("kappa", sort=True):
        condition_sds = group.groupby("topology").final_activated_proportion.std(ddof=1)
        topology_means = group.groupby("topology").final_activated_proportion.mean()
        seed_by_topology = group.pivot(
            index="random_seed",
            columns="topology",
            values="final_activated_proportion",
        )
        paired_topology_sds = seed_by_topology.std(axis=1, ddof=1)
        pooled_seed_sd = float(np.sqrt(np.mean(np.square(condition_sds))))
        topology_mean_sd = float(topology_means.std(ddof=1))
        rows.append(
            {
                "kappa": float(kappa),
                "pooled_within_condition_seed_sd": pooled_seed_sd,
                "pooled_within_condition_seed_variance": pooled_seed_sd**2,
                "sd_of_four_topology_means": topology_mean_sd,
                "variance_of_four_topology_means": topology_mean_sd**2,
                "topology_mean_range": float(topology_means.max() - topology_means.min()),
                "rms_within_seed_topology_sd": float(
                    np.sqrt(np.mean(np.square(paired_topology_sds)))
                ),
                "systematic_topology_signal_to_seed_noise_ratio": (
                    topology_mean_sd / pooled_seed_sd if pooled_seed_sd else math.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def kappa_effects(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_rows = []
    summary_rows = []
    kappas = sorted(data.kappa.unique())
    transitions = list(zip(kappas[:-1], kappas[1:]))
    for topology in TOPOLOGIES:
        subset = data[data.topology == topology]
        wide = subset.pivot(
            index="random_seed",
            columns="kappa",
            values="final_activated_proportion",
        )
        condition_sds = subset.groupby("kappa").final_activated_proportion.std(ddof=1)
        for lower, upper in transitions:
            differences = (wide[upper] - wide[lower]).to_numpy(float)
            pooled_seed_sd = float(
                np.sqrt((condition_sds[lower] ** 2 + condition_sds[upper] ** 2) / 2)
            )
            for seed, value in zip(wide.index, differences):
                raw_rows.append(
                    {
                        "topology": topology,
                        "kappa_lower": float(lower),
                        "kappa_upper": float(upper),
                        "transition": f"{lower:.1f} to {upper:.1f}",
                        "random_seed": int(seed),
                        "paired_difference": float(value),
                    }
                )
            summary_rows.append(
                {
                    "topology": topology,
                    "kappa_lower": float(lower),
                    "kappa_upper": float(upper),
                    "transition": f"{lower:.1f} to {upper:.1f}",
                    **paired_summary(differences),
                    "pooled_seed_sd_at_endpoints": pooled_seed_sd,
                    "absolute_mean_effect_to_seed_sd_ratio": (
                        abs(float(differences.mean())) / pooled_seed_sd
                        if pooled_seed_sd
                        else math.nan
                    ),
                }
            )
    return pd.DataFrame(raw_rows), pd.DataFrame(summary_rows)


def repeated_measures_anova(data: pd.DataFrame) -> pd.DataFrame:
    fit = AnovaRM(
        data,
        depvar="final_activated_proportion",
        subject="random_seed",
        within=["topology", "kappa"],
    ).fit()
    table = fit.anova_table.reset_index().rename(
        columns={
            "index": "effect",
            "F Value": "f_value",
            "Num DF": "df_numerator",
            "Den DF": "df_denominator",
            "Pr > F": "p_value",
        }
    )
    table["partial_eta_squared"] = (
        table.f_value * table.df_numerator
    ) / (table.f_value * table.df_numerator + table.df_denominator)
    table["model"] = (
        "two-factor repeated-measures ANOVA; random_seed is subject/block; "
        "topology and kappa are within-seed categorical factors"
    )
    return table


def contrast_interaction_tests(raw_differences: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for contrast, group in raw_differences.groupby("contrast", sort=False):
        fit = AnovaRM(
            group,
            depvar="paired_difference",
            subject="random_seed",
            within=["kappa"],
        ).fit()
        result = fit.anova_table.iloc[0]
        f_value = float(result["F Value"])
        df_num = float(result["Num DF"])
        df_den = float(result["Den DF"])
        rows.append(
            {
                "contrast": contrast,
                "effect": "change in topology contrast across kappa",
                "f_value": f_value,
                "df_numerator": df_num,
                "df_denominator": df_den,
                "p_value": float(result["Pr > F"]),
                "partial_eta_squared": (f_value * df_num)
                / (f_value * df_num + df_den),
            }
        )
    return pd.DataFrame(rows)


def difference_in_differences(raw_differences: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for contrast, group in raw_differences.groupby("contrast", sort=False):
        wide = group.pivot(
            index="random_seed", columns="kappa", values="paired_difference"
        )
        values = (wide[0.7] - wide[0.4]).to_numpy(float)
        rows.append(
            {
                "contrast": contrast,
                "comparison": "contrast at kappa 0.7 minus contrast at kappa 0.4",
                **paired_summary(values),
            }
        )
    return pd.DataFrame(rows)


def plot_final_distributions(data: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True)
    kappas = sorted(data.kappa.unique())
    for axis, topology in zip(axes.ravel(), TOPOLOGIES):
        subset = data[data.topology == topology]
        values = [
            subset[subset.kappa == kappa].final_activated_proportion.to_numpy()
            for kappa in kappas
        ]
        boxes = axis.boxplot(values, positions=range(len(kappas)), widths=0.65, patch_artist=True)
        for box in boxes["boxes"]:
            box.set_facecolor(COLORS[topology])
            box.set_alpha(0.5)
        axis.set_title(topology, weight="bold")
        axis.set_xticks(range(len(kappas)), [f"{value:.1f}" for value in kappas])
        axis.set_ylim(0, 1.03)
        axis.grid(axis="y", alpha=0.25)
    figure.supxlabel("Kappa")
    figure.supylabel("Final activated proportion")
    figure.suptitle("Figure A — Final Cascade Distributions (20 paired seeds)", weight="bold", fontsize=16)
    figure.tight_layout(rect=(0.02, 0.02, 1, 0.96))
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_paired_differences(raw: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True, sharey=True)
    kappas = sorted(raw.kappa.unique())
    rng = np.random.default_rng(20260921)
    for axis, contrast in zip(axes, [f"{left} - {right}" for left, right in CONTRASTS]):
        subset = raw[raw.contrast == contrast]
        values = [
            subset[subset.kappa == kappa].paired_difference.to_numpy()
            for kappa in kappas
        ]
        axis.boxplot(values, positions=range(len(kappas)), widths=0.58, showfliers=False)
        for position, group in enumerate(values):
            jitter = rng.uniform(-0.16, 0.16, size=len(group))
            axis.scatter(position + jitter, group, s=18, alpha=0.52, color="#355c7d")
        axis.axhline(0, color="black", lw=1.2, ls="--")
        axis.set_title(contrast, weight="bold")
        axis.set_ylabel("Paired final-size difference")
        axis.grid(axis="y", alpha=0.25)
    axes[-1].set_xticks(range(len(kappas)), [f"{value:.1f}" for value in kappas])
    axes[-1].set_xlabel("Kappa")
    figure.suptitle("Figure B — Paired Topology Differences by Kappa", weight="bold", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_kappa_effect_vs_seed_noise(summary: pd.DataFrame, path: Path) -> None:
    focus = summary[
        summary.apply(
            lambda row: (row.kappa_lower, row.kappa_upper) in FOCUS_TRANSITIONS,
            axis=1,
        )
    ]
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    labels = [f"{lower:.1f}→{upper:.1f}" for lower, upper in FOCUS_TRANSITIONS]
    for axis, topology in zip(axes.ravel(), TOPOLOGIES):
        group = focus[focus.topology == topology].set_index("transition")
        ordered = group.loc[[f"{a:.1f} to {b:.1f}" for a, b in FOCUS_TRANSITIONS]]
        x = np.arange(len(ordered))
        mean = ordered.mean_difference.to_numpy(float)
        lower_error = mean - ordered.ci95_lower.to_numpy(float)
        upper_error = ordered.ci95_upper.to_numpy(float) - mean
        axis.errorbar(
            x,
            mean,
            yerr=np.vstack([lower_error, upper_error]),
            fmt="o-",
            capsize=5,
            lw=2,
            color=COLORS[topology],
            label="Mean paired kappa effect (95% CI)",
        )
        axis.plot(
            x,
            ordered.pooled_seed_sd_at_endpoints,
            "D--",
            color="#555555",
            label="Pooled seed SD at endpoints",
        )
        axis.axhline(0, color="black", lw=1)
        axis.set_title(topology, weight="bold")
        axis.set_xticks(x, labels)
        axis.set_ylabel("Activated-proportion scale")
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(fontsize=9)
    figure.suptitle("Figure C — Kappa Effect Signal versus Seed Variation", weight="bold", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_report(
    topology_summary: pd.DataFrame,
    variation: pd.DataFrame,
    kappa_summary: pd.DataFrame,
    anova: pd.DataFrame,
    contrast_tests: pd.DataFrame,
    did: pd.DataFrame,
    seed_grand_means: pd.DataFrame,
    path: Path,
) -> None:
    focus = kappa_summary[
        kappa_summary.apply(
            lambda row: (row.kappa_lower, row.kappa_upper) in FOCUS_TRANSITIONS,
            axis=1,
        )
    ]
    seed_mean_sd = float(seed_grand_means.seed_grand_mean.std(ddof=1))
    seed_mean_range = float(
        seed_grand_means.seed_grand_mean.max() - seed_grand_means.seed_grand_mean.min()
    )
    lines = [
        "# Paired topology × kappa statistical analysis",
        "",
        "This analysis reads the existing `raw_run_results.csv`; no simulation was rerun.",
        "",
        "## Statistical design",
        "",
        "The omnibus model is a balanced two-factor repeated-measures ANOVA with random seed as the subject/block and topology (4 levels) and kappa (11 categorical levels) as within-seed factors. Partial eta-squared is reported as the omnibus effect-size measure. Paired topology and adjacent-kappa contrasts use paired t confidence intervals, Cohen's dz, and direction proportions. Because final cascade sizes are often bimodal, paired distributions and effect magnitudes are treated as primary evidence; ANOVA p-values are supporting summaries and rely on the usual repeated-measures assumptions.",
        "",
        "## Conservative conclusions",
        "",
        "- **Kappa effect:** Strong overall and usually positive through the 0.4–0.7 transition region, but its absolute paired mean is generally smaller than the raw cross-seed SD. Pairing is therefore essential.",
        "- **Topology effect:** There is no stable topology ordering across all kappa values. The omnibus topology main effect is modest and does not provide strong evidence for one topology being uniformly higher.",
        "- **Seed variation:** Larger than the systematic between-topology mean variation at every kappa. The SD of seed-level grand means across all 44 cells is "
        f"{seed_mean_sd:.4f}, with a range of {seed_mean_range:.4f}.",
        "- **Topology by kappa interaction:** Supported by the omnibus repeated-measures interaction. It is strongest for WS versus ER, present but seed-sensitive for BA versus ER, and not evident for SBM versus ER.",
        "- **Intervention region:** Kappa 0.4, 0.5, and 0.6 remain suitable because they span the transition and contain the largest topology separation. Future comparisons should retain paired seeds and continuous cascade size because stochastic variation is substantial.",
        "",
        "## Omnibus repeated-measures ANOVA",
        "",
        anova.to_markdown(index=False, floatfmt=".6g"),
        "",
        "## Topology contrast changes across kappa",
        "",
        contrast_tests.to_markdown(index=False, floatfmt=".6g"),
        "",
        "## Kappa 0.4 to 0.7 difference-in-differences",
        "",
        did.to_markdown(index=False, floatfmt=".6g"),
        "",
        "## Seed noise versus systematic topology signal",
        "",
        variation.to_markdown(index=False, floatfmt=".5f"),
        "",
        "## Focus-region adjacent kappa effects",
        "",
        focus.to_markdown(index=False, floatfmt=".5f"),
        "",
        "## Full paired topology summaries",
        "",
        topology_summary.to_markdown(index=False, floatfmt=".5f"),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    data = pd.read_csv(INPUT_PATH)
    validate_balanced_design(data)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    topology_raw, topology_summary = topology_differences(data)
    variation = seed_vs_topology_variation(data)
    kappa_raw, kappa_summary = kappa_effects(data)
    anova = repeated_measures_anova(data)
    contrast_tests = contrast_interaction_tests(topology_raw)
    did = difference_in_differences(topology_raw)
    seed_grand_means = (
        data.groupby("random_seed", as_index=False).final_activated_proportion.mean()
        .rename(columns={"final_activated_proportion": "seed_grand_mean"})
    )

    topology_raw.to_csv(OUTPUT_DIR / "paired_topology_differences_raw.csv", index=False)
    topology_summary.to_csv(OUTPUT_DIR / "paired_topology_differences_summary.csv", index=False)
    variation.to_csv(OUTPUT_DIR / "seed_vs_topology_variation.csv", index=False)
    kappa_raw.to_csv(OUTPUT_DIR / "paired_kappa_effects_raw.csv", index=False)
    kappa_summary.to_csv(OUTPUT_DIR / "paired_kappa_effects_summary.csv", index=False)
    anova.to_csv(OUTPUT_DIR / "repeated_measures_anova.csv", index=False)
    contrast_tests.to_csv(OUTPUT_DIR / "topology_contrast_interaction_tests.csv", index=False)
    did.to_csv(OUTPUT_DIR / "interaction_difference_in_differences.csv", index=False)
    seed_grand_means.to_csv(OUTPUT_DIR / "seed_grand_means.csv", index=False)

    plot_final_distributions(data, OUTPUT_DIR / "figure_A_final_size_distributions.png")
    plot_paired_differences(topology_raw, OUTPUT_DIR / "figure_B_paired_topology_differences.png")
    plot_kappa_effect_vs_seed_noise(kappa_summary, OUTPUT_DIR / "figure_C_kappa_effect_vs_seed_variation.png")
    write_report(
        topology_summary,
        variation,
        kappa_summary,
        anova,
        contrast_tests,
        did,
        seed_grand_means,
        OUTPUT_DIR / "analysis_report.md",
    )

    print(f"Read {len(data)} existing run rows; no simulation was executed")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    print("Repeated-measures ANOVA:")
    print(anova.to_string(index=False))


if __name__ == "__main__":
    main()
