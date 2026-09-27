"""Observable early-cascade features and a small reproducible logistic model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats

from baseline_simulation import B, R
from intervention import (
    ExposurePaths,
    apply_decision_rule,
    build_exposure_paths,
    compute_local_exposure,
)


OBSERVATION_TIMESTEP = 5
TOPOLOGY_REFERENCE = "ER"
TOPOLOGY_INDICATORS = ("WS", "BA", "SBM")


def observe_baseline_cascade(
    simulation,
    kappa: float,
    ignition_nodes: np.ndarray,
    *,
    observation_timestep: int = OBSERVATION_TIMESTEP,
    max_steps: int = 60,
    prepared_paths: ExposurePaths | None = None,
) -> tuple[dict[str, float], np.ndarray]:
    """Run the unchanged baseline and return features available by one time."""

    paths = prepared_paths or build_exposure_paths(simulation)
    state = np.zeros(simulation.n, dtype=int)
    state[np.asarray(ignition_nodes, dtype=int)] = 1
    states = [state.copy()]
    counts = [int(state.sum())]
    stable = False
    for _ in range(1, max_steps + 1):
        if stable:
            new_state = state.copy()
        else:
            local = compute_local_exposure(state, paths, np.ones(paths.count, dtype=float))
            new_state = apply_decision_rule(simulation, state, local, kappa, B, R)
            stable = np.array_equal(new_state, state)
        state = new_state
        states.append(state.copy())
        counts.append(int(state.sum()))

    observed = states[observation_timestep]
    previous = states[observation_timestep - 1]
    proportions = np.asarray(counts[: observation_timestep + 1], dtype=float) / simulation.n
    increments = np.diff(proportions)
    degrees = paths.original_degrees
    degree_scale = float(degrees.max()) if degrees.size and degrees.max() > 0 else 1.0
    active = observed == 1
    inactive = ~active
    newly_active = active & (previous == 0)
    frontier = active[paths.sources] & inactive[paths.targets]
    local = compute_local_exposure(observed, paths, np.ones(paths.count, dtype=float))

    features: dict[str, float] = {"kappa": float(kappa)}
    for timestep in range(1, observation_timestep + 1):
        features[f"activated_t{timestep}"] = float(proportions[timestep])
        features[f"increment_t{timestep}"] = float(increments[timestep - 1])
    features["last_acceleration"] = float(increments[-1] - increments[-2])
    features["frontier_path_fraction"] = float(frontier.mean())
    features["frontier_paths_per_inactive"] = float(frontier.sum() / max(1, inactive.sum()))
    features["mean_local_exposure_inactive"] = float(local[inactive].mean()) if inactive.any() else 0.0
    features["p90_local_exposure_inactive"] = float(np.quantile(local[inactive], 0.9)) if inactive.any() else 0.0
    features["active_mean_degree_norm"] = float(degrees[active].mean() / degree_scale) if active.any() else 0.0
    features["newly_active_mean_degree_norm"] = float(degrees[newly_active].mean() / degree_scale) if newly_active.any() else 0.0
    topology = simulation.graph.graph.get("topology", TOPOLOGY_REFERENCE)
    for name in TOPOLOGY_INDICATORS:
        features[f"topology_{name}"] = float(topology == name)
    return features, np.asarray(counts, dtype=int)


def roc_auc(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    positives = int(y_true.sum())
    negatives = int(y_true.size - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = stats.rankdata(probabilities)
    positive_rank_sum = ranks[y_true == 1].sum()
    return float((positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


@dataclass
class LogisticRiskModel:
    feature_names: tuple[str, ...]
    means: np.ndarray
    scales: np.ndarray
    coefficients: np.ndarray
    intercept: float
    l2: float
    threshold: float = 0.5

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        standardized = (np.asarray(matrix, dtype=float) - self.means) / self.scales
        logits = self.intercept + standardized @ self.coefficients
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        return (self.predict_proba(matrix) >= self.threshold).astype(int)

    def to_dict(self) -> dict:
        return {
            "feature_names": list(self.feature_names),
            "means": self.means.tolist(),
            "scales": self.scales.tolist(),
            "coefficients": self.coefficients.tolist(),
            "intercept": self.intercept,
            "l2": self.l2,
            "threshold": self.threshold,
        }


def fit_logistic_risk_model(
    matrix: np.ndarray,
    labels: np.ndarray,
    feature_names: tuple[str, ...],
    *,
    l2: float,
) -> LogisticRiskModel:
    matrix = np.asarray(matrix, dtype=float)
    labels = np.asarray(labels, dtype=float)
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0, ddof=0)
    scales[scales < 1e-12] = 1.0
    standardized = (matrix - means) / scales

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        intercept = parameters[0]
        coefficients = parameters[1:]
        logits = intercept + standardized @ coefficients
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
        loss = np.mean(np.logaddexp(0.0, logits) - labels * logits)
        loss += 0.5 * l2 * np.dot(coefficients, coefficients) / labels.size
        error = probabilities - labels
        gradient = np.concatenate((
            [error.mean()],
            standardized.T @ error / labels.size + l2 * coefficients / labels.size,
        ))
        return float(loss), gradient

    fitted = optimize.minimize(
        objective,
        np.zeros(matrix.shape[1] + 1),
        method="L-BFGS-B",
        jac=True,
    )
    if not fitted.success:
        raise RuntimeError(f"Logistic optimization failed: {fitted.message}")
    return LogisticRiskModel(
        feature_names=feature_names,
        means=means,
        scales=scales,
        coefficients=fitted.x[1:],
        intercept=float(fitted.x[0]),
        l2=float(l2),
    )


def classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    predictions = probabilities >= threshold
    tp = int(np.sum(predictions & (labels == 1)))
    fp = int(np.sum(predictions & (labels == 0)))
    tn = int(np.sum(~predictions & (labels == 0)))
    fn = int(np.sum(~predictions & (labels == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    beta2 = 2.0
    f2 = (1 + beta2**2) * precision * recall / (beta2**2 * precision + recall) if precision + recall else 0.0
    return {
        "n": int(labels.size),
        "prevalence": float(labels.mean()),
        "auc": roc_auc(labels, probabilities),
        "brier": float(np.mean((probabilities - labels) ** 2)),
        "accuracy": float(np.mean(predictions == labels)),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "f2": f2,
        "intervention_rate": float(predictions.mean()),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def select_f2_threshold(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, dict[str, float]]:
    candidates = np.unique(np.concatenate(([0.0, 1.0], probabilities)))
    best_threshold = 0.5
    best_metrics: dict[str, float] | None = None
    for threshold in candidates:
        metrics = classification_metrics(labels, probabilities, float(threshold))
        score = (metrics["f2"], metrics["specificity"], threshold)
        if best_metrics is None or score > (
            best_metrics["f2"], best_metrics["specificity"], best_threshold
        ):
            best_threshold = float(threshold)
            best_metrics = metrics
    assert best_metrics is not None
    return best_threshold, best_metrics
