"""Run a small synthetic epsilon-insensitive calibration demonstration.

This is intentionally outside the default test suite. It gives reviewers a
cheap way to compare grouped relative loss against the proposed
epsilon-insensitive policy loss on a fixture with high-cardinality soft targets
and a small set of hard population anchors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse as sp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="epsilon_calibration_demo_summary.json",
        help="JSON summary path.",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    try:
        from l0.calibration import SparseCalibrationWeights
    except ImportError as exc:
        raise SystemExit(
            "Install the l0 extra or pin the L0 feature branch before running "
            "this demo."
        ) from exc

    args = parse_args()
    rng = np.random.default_rng(args.seed)
    n_features = 160
    n_soft = 80
    n_targets = 2 + n_soft

    weights_true = rng.lognormal(mean=1.0, sigma=0.5, size=n_features)
    matrix = rng.gamma(shape=2.0, scale=1.0, size=(n_targets, n_features))
    matrix[0, :] = 1.0
    matrix[1, :] = rng.uniform(0.0, 1.0, size=n_features) > 0.45
    targets = matrix @ weights_true

    target_groups = np.array([0, 1] + [2] * n_soft)
    initial_weights = np.full(n_features, targets[0] / n_features)
    M = sp.csr_matrix(matrix)

    grouped = SparseCalibrationWeights(
        n_features=n_features,
        init_keep_prob=0.95,
        init_weights=initial_weights,
        log_weight_jitter_sd=0.05,
        seed=args.seed,
    )
    grouped.fit(
        M,
        targets,
        lambda_l0=1e-3,
        lr=0.1,
        epochs=args.epochs,
        loss_type="relative",
        target_groups=target_groups,
        verbose=False,
    )

    epsilon = SparseCalibrationWeights(
        n_features=n_features,
        init_keep_prob=0.95,
        init_weights=initial_weights,
        log_weight_jitter_sd=0.05,
        seed=args.seed,
    )
    target_weights = np.array([40.0, 40.0] + [2.0] * n_soft)
    target_tolerances = np.array([0.005, 0.005] + [0.10] * n_soft)
    target_scales = np.maximum(
        np.abs(targets), np.array([100.0, 100.0] + [1.0] * n_soft)
    )
    epsilon.fit(
        M,
        targets,
        lambda_l0=1e-3,
        lr=0.1,
        epochs=args.epochs,
        loss_type="relative_epsilon",
        target_weights=target_weights,
        target_tolerances=target_tolerances,
        target_scales=target_scales,
        verbose=False,
    )

    result = {
        "grouped_relative": _summarize(grouped, M, targets, target_tolerances),
        "epsilon_insensitive": _summarize(epsilon, M, targets, target_tolerances),
        "hard_anchor_tolerance": 0.005,
        "soft_target_tolerance": 0.10,
        "epochs": args.epochs,
        "seed": args.seed,
    }
    output_path = Path(args.output)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {output_path}")


def _summarize(model, M, targets: np.ndarray, tolerances: np.ndarray) -> dict:
    prediction = model.predict(M).detach().cpu().numpy()
    rel_errors = np.abs((prediction - targets) / np.maximum(np.abs(targets), 1.0))
    weights = model.get_weights(deterministic=True).detach().cpu().numpy()
    return {
        "hard_anchor_max_abs_rel_error": float(rel_errors[:2].max()),
        "soft_target_mean_abs_rel_error": float(rel_errors[2:].mean()),
        "share_targets_within_tolerance": float((rel_errors <= tolerances).mean()),
        "weight_sum": float(weights.sum()),
        "nonzero_weights": int((weights > 0).sum()),
    }


if __name__ == "__main__":
    main()
