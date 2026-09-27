"""Cancel-culture baseline simulation with selectable network topology.

The default invocation retains the original WS baseline and experiment sweep.
Only graph construction is configurable; agent calibration, KOL setup, seeding,
utility thresholds, kappa exposure, and cascade decisions are unchanged.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy import stats

from network_generation import SUPPORTED_TOPOLOGIES, generate_network


# Original baseline parameters.
SURVEY_PATH = Path(__file__).with_name("群體輿論如何影響個人發言意願 (回覆) (1).xlsx")
N = 10_000
K = 10
P = 0.1
NUM_KOLS = 20
KOL_REACH = 200
NUM_IGNITE = 5
B = 1.5
R = 0
RANDOM_SEED = 42


def parse_pct(val: object) -> float:
    """Convert numeric proportions or percentage strings to numbers in [0, 1]."""

    if isinstance(val, Real) and not isinstance(val, bool):
        numeric = float(val)
        if np.isnan(numeric):
            return np.nan
        if 0 <= numeric <= 1:
            return numeric
        if 1 < numeric <= 100:
            return numeric / 100
        raise ValueError(f"Percentage value out of range: {val!r}")

    value = str(val).strip()
    if not value or value.lower() == "nan":
        return np.nan
    has_percent_sign = "%" in value
    value = value.replace("%", "").strip()
    if "~" in value:
        lower, upper = value.split("~")
        midpoint = (float(lower) + float(upper)) / 2
        return midpoint / 100 if has_percent_sign or midpoint > 1 else midpoint
    numeric = float(value)
    if has_percent_sign or numeric > 1:
        numeric /= 100
    if not 0 <= numeric <= 1:
        raise ValueError(f"Percentage value out of range: {val!r}")
    return numeric


def load_survey_and_fit(path: str | Path):
    """Read the existing survey and fit the original Beta distributions."""

    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        dataframe = pd.read_excel(path)
    else:
        dataframe = pd.read_csv(path)

    q1_col, q4_col, q5_col = dataframe.columns[1], dataframe.columns[7], dataframe.columns[8]
    dataframe["is_poster"] = dataframe[q1_col].str.startswith("是", na=False)
    q4_raw = dataframe.loc[dataframe["is_poster"], q4_col].apply(parse_pct).dropna()
    q5_raw = dataframe.loc[dataframe["is_poster"], q5_col].apply(parse_pct).dropna()
    poster_ratio = dataframe["is_poster"].sum() / len(dataframe)

    epsilon = 1e-4
    a4, b4, _, _ = stats.beta.fit(q4_raw.clip(epsilon, 1 - epsilon), floc=0, fscale=1)
    a5, b5, _, _ = stats.beta.fit(q5_raw.clip(epsilon, 1 - epsilon), floc=0, fscale=1)
    return poster_ratio, (a4, b4), (a5, b5)


@dataclass
class BaselineSimulation:
    """State for the original agent model running on a supplied graph."""

    graph: object
    base_graph: object
    roles: np.ndarray
    c_private: np.ndarray
    c_public: np.ndarray
    kol_idx: np.ndarray
    neighbors: list[np.ndarray]
    rng: np.random.Generator

    @property
    def n(self) -> int:
        return len(self.roles)

    def step(self, state: np.ndarray, kappa: float, benefit: float, outside_payoff: float) -> np.ndarray:
        """Apply the original deterministic, one-way agent decision rule."""

        phi_local = np.zeros(self.n)
        for node in range(self.n):
            if len(self.neighbors[node]) > 0:
                phi_local[node] = state[self.neighbors[node]].mean()

        poster_mask = self.roles == "Poster"
        phi_global = state[poster_mask].mean() if poster_mask.sum() > 0 else 0.0
        new_state = state.copy()

        for node in range(self.n):
            if state[node] == 1:
                continue
            if self.roles[node] == "Lurker":
                effective_threshold = (self.c_private[node] + outside_payoff) / benefit
                if phi_local[node] >= effective_threshold:
                    new_state[node] = 1
            else:
                effective_threshold = (self.c_public[node] + outside_payoff) / benefit
                phi_perceived = (1 - kappa) * phi_local[node] + kappa * phi_global
                if phi_perceived >= effective_threshold:
                    new_state[node] = 1

        return new_state

    def run(
        self,
        kappa: float,
        benefit: float = B,
        outside_payoff: float = R,
        max_steps: int = 60,
        num_ignite: int = NUM_IGNITE,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run one complete cascade with the original KOL seeding rule."""

        state = np.zeros(self.n, dtype=int)
        ignite_kols = self.rng.choice(self.kol_idx, size=num_ignite, replace=False)
        state[ignite_kols] = 1

        history = [state.mean()]
        for _ in range(max_steps):
            new_state = self.step(state, kappa, benefit, outside_payoff)
            history.append(new_state.mean())
            if np.array_equal(new_state, state):
                break
            state = new_state
        return np.asarray(history), state


def create_simulation(
    topology: str = "WS",
    n: int = N,
    target_avg_degree: float = K,
    network_seed: int = RANDOM_SEED,
    agent_seed: int = RANDOM_SEED,
    survey_path: str | Path = SURVEY_PATH,
    **topology_params,
) -> BaselineSimulation:
    """Create the unchanged baseline agent model on a selected topology."""

    poster_ratio, beta_private, beta_public = load_survey_and_fit(survey_path)
    rng = np.random.default_rng(agent_seed)
    n_posters = round(n * poster_ratio)
    n_lurkers = n - n_posters

    roles = np.array(["Poster"] * n_posters + ["Lurker"] * n_lurkers)
    rng.shuffle(roles)
    c_private = rng.beta(beta_private[0], beta_private[1], size=n)
    c_public = np.full(n, np.nan)
    poster_idx = np.where(roles == "Poster")[0]
    c_public[poster_idx] = rng.beta(beta_public[0], beta_public[1], size=n_posters)

    resolved_topology_params = dict(topology_params)
    if topology.upper() == "WS":
        resolved_topology_params.setdefault("p", P)
    base_graph = generate_network(
        topology,
        n=n,
        target_avg_degree=target_avg_degree,
        seed=network_seed,
        **resolved_topology_params,
    )
    graph = base_graph.copy()

    # Original KOL setup and random-number order.
    kol_idx = rng.choice(n, size=NUM_KOLS, replace=False)
    roles[kol_idx] = "Poster"
    c_public[kol_idx] = rng.uniform(abs(R) + 0.02, 0.50, size=NUM_KOLS)
    for kol in kol_idx:
        target_nodes = rng.choice(n, size=KOL_REACH, replace=False)
        for target in target_nodes:
            if kol != target:
                graph.add_edge(int(kol), int(target))

    neighbors = [np.fromiter(graph.neighbors(node), dtype=int) for node in range(n)]
    return BaselineSimulation(
        graph=graph,
        base_graph=base_graph,
        roles=roles,
        c_private=c_private,
        c_public=c_public,
        kol_idx=kol_idx,
        neighbors=neighbors,
        rng=rng,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", choices=SUPPORTED_TOPOLOGIES, default="WS")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--kappa", type=float, default=0.6)
    parser.add_argument("--single-run", action="store_true", help="Run one cascade instead of the legacy kappa sweep")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print(f"[{time.strftime('%H:%M:%S')}] Fitting the survey threshold distributions...")
    print(
        f"[{time.strftime('%H:%M:%S')}] Building {args.topology} network "
        f"(N={N}, target average degree={K}, seed={args.seed})..."
    )
    simulation = create_simulation(topology=args.topology, network_seed=args.seed, agent_seed=args.seed)

    history_single, final_state = simulation.run(kappa=args.kappa)
    print(f"  Final activated proportion ({args.topology}, kappa={args.kappa:.2f}): {final_state.mean():.1%}")
    if args.single_run:
        return

    kappa_range = np.linspace(0.0, 1.0, 15)
    finals = []
    for kappa in kappa_range:
        _, state = simulation.run(kappa=kappa)
        finals.append(state.mean())
        print(f"  kappa={kappa:.2f} -> activated proportion: {state.mean():.1%}")

    if args.no_plot:
        return

    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.rcParams["font.sans-serif"] = [
        "PingFang TC",
        "Heiti TC",
        "Arial Unicode MS",
        "Microsoft JhengHei",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False

    figure, (axis_history, axis_kappa) = plt.subplots(1, 2, figsize=(14, 5))
    axis_history.plot(history_single, "-o", color="steelblue", lw=2)
    axis_history.set_title(f"A. Cascade time series ({args.topology}, kappa={args.kappa:.1f})")
    axis_history.set_xlabel("Time Step")
    axis_history.set_ylabel("Activated proportion")
    axis_history.set_ylim(0, 1)
    axis_history.grid(alpha=0.3)

    axis_kappa.plot(kappa_range, finals, "-o", color="crimson", lw=2)
    axis_kappa.set_title(f"B. Kappa vs final cascade size ({args.topology})")
    axis_kappa.set_xlabel("Algorithmic exposure strength (kappa)")
    axis_kappa.set_ylabel("Final activated proportion")
    axis_kappa.set_ylim(0, 1)
    axis_kappa.fill_between(kappa_range, 0, finals, color="crimson", alpha=0.1)
    axis_kappa.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
