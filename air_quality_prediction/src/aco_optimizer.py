"""
Ant Colony Optimization (ACO) for feature selection in air-quality prediction.

The search is multi-objective:
1. Prediction accuracy: RMSE of a Decision Tree on a held-out validation set.
2. Interpretability for policy: prefer features derived from pollutants that
   emission controls can act on (PM2.5, NO2, SO2, ...) over weather features.

Feature names are matched by their base variable, so engineered columns such
as ``PM2.5_lag1`` or ``NO2_avg7`` count as policy-controllable and
``temperature`` or ``wind_speed`` count as uncontrollable.
"""

import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

logger = logging.getLogger(__name__)

POLICY_CONTROLLABLE = [
    "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "SO2", "CO", "O3",
    "Benzene", "Toluene", "Xylene", "PM_ratio",
]
UNCONTROLLABLE = [
    "temperature", "humidity", "wind_speed", "wind_direction", "pressure",
    "precipitation",
]


def feature_category(name: str) -> str:
    """Return 'policy', 'weather' or 'other' for a (possibly engineered) feature."""
    base = name.split("_lag")[0].split("_avg")[0]
    if base in POLICY_CONTROLLABLE or name in POLICY_CONTROLLABLE:
        return "policy"
    if base.lower() in UNCONTROLLABLE:
        return "weather"
    return "other"


class ACO_FeatureSelection:
    """
    Ant Colony Optimization for feature-subset selection with a policy bias.

    Fitness (lower is better), scale-free so it works for any target unit::

        fitness = val_rmse * (1 + size_penalty * k / n
                                - policy_bonus * share_policy
                                + weather_penalty * share_weather)

    where k is the subset size, n the number of candidate features and the
    shares are fractions of the subset.
    """

    def __init__(self,
                 n_ants: int = 30,
                 n_iterations: int = 60,
                 alpha: float = 1.0,
                 beta: float = 2.0,
                 rho: float = 0.1,
                 q0: float = 0.5,
                 policy_weight: float = 1.5,
                 size_penalty: float = 0.05,
                 policy_bonus: float = 0.03,
                 weather_penalty: float = 0.03,
                 min_features: int = 3,
                 max_features: int = 15,
                 patience: int = 15,
                 tree_params: Optional[Dict] = None,
                 random_state: Optional[int] = 42):
        """
        Args:
            n_ants: Ants (candidate subsets) per iteration.
            n_iterations: Maximum iterations.
            alpha: Pheromone importance.
            beta: Heuristic importance.
            rho: Pheromone evaporation rate.
            q0: Probability of greedy choice at each step (else roulette).
            policy_weight: Heuristic multiplier for policy features
                (weather features get 1 / policy_weight).
            size_penalty, policy_bonus, weather_penalty: Fitness terms, see class doc.
            min_features, max_features: Subset size range (inclusive).
            patience: Stop after this many iterations without improvement.
            tree_params: Parameters of the Decision Tree used for evaluation.
            random_state: Seed for reproducibility.
        """
        if min_features < 1 or max_features < min_features:
            raise ValueError("Need 1 <= min_features <= max_features")
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q0 = q0
        self.policy_weight = policy_weight
        self.size_penalty = size_penalty
        self.policy_bonus = policy_bonus
        self.weather_penalty = weather_penalty
        self.min_features = min_features
        self.max_features = max_features
        self.patience = patience
        self.tree_params = tree_params or dict(
            max_depth=10, min_samples_split=20, min_samples_leaf=10)
        self.random_state = random_state
        self.policy_controllable = POLICY_CONTROLLABLE
        self.uncontrollable = UNCONTROLLABLE
        self._reset()

    # ------------------------------------------------------------------ state
    def _reset(self):
        self.rng = np.random.default_rng(self.random_state)
        self.best_solution: Optional[List[str]] = None
        self.best_fitness = float("inf")
        self.best_val_rmse = float("inf")
        self.convergence_history: List[Dict] = []
        self.feature_selection_history: List[Dict] = []
        self._cache: Dict[frozenset, Tuple[float, float]] = {}

    # -------------------------------------------------------------- heuristic
    def _calculate_heuristic(self, features: Sequence[str],
                             target_correlation: Dict[str, float]) -> np.ndarray:
        h = []
        for f in features:
            base = abs(target_correlation.get(f, 0.0))
            cat = feature_category(f)
            bias = (self.policy_weight if cat == "policy"
                    else 1.0 / self.policy_weight if cat == "weather" else 1.0)
            h.append(base * bias)
        h = np.asarray(h, dtype=float)
        return np.maximum(h, 1e-6)  # every feature stays reachable

    # ------------------------------------------------------------ solutions
    def _construct_solution(self, n: int, tau: np.ndarray, eta: np.ndarray) -> List[int]:
        k = int(self.rng.integers(self.min_features, min(self.max_features, n) + 1))
        available = np.ones(n, dtype=bool)
        chosen: List[int] = []
        weights = (tau ** self.alpha) * (eta ** self.beta)
        for _ in range(k):
            w = np.where(available, weights, 0.0)
            if self.rng.random() < self.q0:
                idx = int(np.argmax(w))
            else:
                idx = int(self.rng.choice(n, p=w / w.sum()))
            chosen.append(idx)
            available[idx] = False
        return chosen

    def _evaluate(self, subset: List[str], X_train, y_train, X_val, y_val,
                  n_total: int) -> Tuple[float, float]:
        key = frozenset(subset)
        if key in self._cache:
            return self._cache[key]
        dt = DecisionTreeRegressor(random_state=self.random_state, **self.tree_params)
        dt.fit(X_train[subset], y_train)
        rmse = float(np.sqrt(mean_squared_error(y_val, dt.predict(X_val[subset]))))

        cats = [feature_category(f) for f in subset]
        share_policy = cats.count("policy") / len(subset)
        share_weather = cats.count("weather") / len(subset)
        factor = (1 + self.size_penalty * len(subset) / n_total
                  - self.policy_bonus * share_policy
                  + self.weather_penalty * share_weather)
        result = (rmse * factor, rmse)
        self._cache[key] = result
        return result

    # ------------------------------------------------------------ optimize
    def optimize(self, X_train: pd.DataFrame, y_train: pd.Series,
                 X_val: pd.DataFrame, y_val: pd.Series) -> Dict:
        """
        Search for the best feature subset.

        Subsets are scored on (X_val, y_val), which must not overlap the test
        data used to report final results. Missing values are imputed with
        training-set medians.
        """
        self._reset()
        features = X_train.select_dtypes(include=[np.number]).columns.tolist()
        n = len(features)
        if n < self.min_features:
            raise ValueError(f"Only {n} numeric features, min_features={self.min_features}")

        medians = X_train[features].median()
        Xtr = X_train[features].fillna(medians)
        Xva = X_val[features].fillna(medians)

        corr = {f: (Xtr[f].corr(y_train) if Xtr[f].std() > 0 else 0.0) for f in features}
        corr = {f: (0.0 if pd.isna(c) else c) for f, c in corr.items()}
        eta = self._calculate_heuristic(features, corr)
        tau = np.ones(n)
        tau_max, tau_min = 1.0, 0.01  # MAX-MIN bounds avoid stagnation

        logger.info("ACO: %d candidate features, %d ants, %d iterations",
                    n, self.n_ants, self.n_iterations)
        stale = 0
        iteration = 0
        for iteration in range(1, self.n_iterations + 1):
            ants = []
            for _ in range(self.n_ants):
                idx = self._construct_solution(n, tau, eta)
                subset = [features[i] for i in idx]
                fit, rmse = self._evaluate(subset, Xtr, y_train, Xva, y_val, n)
                ants.append((idx, fit, rmse))

            best_idx, best_fit, best_rmse = min(ants, key=lambda a: a[1])
            if best_fit < self.best_fitness - 1e-9:
                self.best_fitness = best_fit
                self.best_val_rmse = best_rmse
                self.best_solution = [features[i] for i in best_idx]
                stale = 0
                logger.info("iter %d: new best, %d features, val RMSE %.3f",
                            iteration, len(best_idx), best_rmse)
            else:
                stale += 1

            # Evaporate, then deposit for the iteration-best and global-best ants
            tau *= (1 - self.rho)
            tau[best_idx] += self.rho * tau_max
            gb = [features.index(f) for f in self.best_solution]
            tau[gb] += self.rho * tau_max * 0.5
            np.clip(tau, tau_min, tau_max, out=tau)

            self.convergence_history.append({
                "iteration": iteration,
                "best_fitness": self.best_fitness,
                "iteration_fitness": best_fit,
                "best_val_rmse": self.best_val_rmse,
                "n_features": len(best_idx),
                "pheromone_mean": float(tau.mean()),
            })
            self.feature_selection_history.append({
                "iteration": iteration,
                "selected_features": [features[i] for i in best_idx],
            })
            if stale >= self.patience:
                logger.info("Early stop: %d iterations without improvement", stale)
                break

        self.pheromones_ = dict(zip(features, tau))
        return {
            "best_features": list(self.best_solution),
            "best_fitness": self.best_fitness,
            "val_rmse": self.best_val_rmse,
            "convergence_history": self.convergence_history,
            "feature_selection_history": self.feature_selection_history,
            "total_iterations": iteration,
            "n_evaluations": len(self._cache),
            "policy_controllable_selected": [f for f in self.best_solution if feature_category(f) == "policy"],
            "uncontrollable_selected": [f for f in self.best_solution if feature_category(f) == "weather"],
        }

    # ------------------------------------------------------------ reporting
    @staticmethod
    def evaluate_model(y_true, y_pred) -> Dict[str, float]:
        return {
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
        }

    def plot_convergence(self, save_path: Optional[str] = None):
        import matplotlib.pyplot as plt
        if not self.convergence_history:
            logger.warning("No convergence history available")
            return None
        h = pd.DataFrame(self.convergence_history)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))
        ax1.plot(h["iteration"], h["best_fitness"], label="Global best", lw=2)
        ax1.plot(h["iteration"], h["iteration_fitness"], "--", alpha=0.7, label="Iteration best")
        ax1.set(xlabel="Iteration", ylabel="Fitness (lower is better)", title="ACO fitness")
        ax1.legend(); ax1.grid(alpha=0.3)
        ax2.plot(h["iteration"], h["n_features"], lw=2)
        ax2.set(xlabel="Iteration", ylabel="Features in iteration-best subset", title="Subset size")
        ax2.grid(alpha=0.3)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def save_results(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "best_features.txt"), "w") as f:
            f.write("ACO Feature Selection Results\n" + "=" * 40 + "\n\n")
            for i, feat in enumerate(self.best_solution or [], 1):
                f.write(f"{i:2d}. {feat} ({feature_category(feat)})\n")
        pd.DataFrame(self.convergence_history).to_csv(
            os.path.join(output_dir, "convergence_history.csv"), index=False)
