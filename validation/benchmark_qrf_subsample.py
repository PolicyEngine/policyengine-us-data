"""Benchmark QRF subsample sizes for memory, time, and accuracy.

Sweeps PUF subsample sizes to find the practical upper bound
for GitHub Actions (~7 GB RAM). Reuses _stratified_subsample_index
and _batch_qrf from puf_impute.py directly.

Usage:
    python validation/benchmark_qrf_subsample.py \
        --sizes 20000 40000 60000 80000 100000 \
        --output validation/outputs/subsample_benchmark.csv \
        --puf-dataset <path> --cps-dataset <path>
"""

import argparse
import gc
import logging
import os
import sys
import threading
import time
import tracemalloc
from typing import Dict, List

import numpy as np
import pandas as pd
import psutil
from scipy import stats

# Allow imports from project root
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)

from policyengine_us_data.calibration.puf_impute import (
    DEMOGRAPHIC_PREDICTORS,
    IMPUTED_VARIABLES,
    _batch_qrf,
    _stratified_subsample_index,
)

logger = logging.getLogger(__name__)

GH_ACTIONS_MEMORY_LIMIT_GB = 7.0
GH_ACTIONS_MEMORY_LIMIT_BYTES = int(GH_ACTIONS_MEMORY_LIMIT_GB * 1024**3)

# Policy-relevant variables (batches 1-2) plus one variable from
# each of batches 3-7 to cover all QRF batch boundaries.
# Batch assignments assume _batch_qrf(..., batch_size=10).
KEY_VARIABLES = [
    # Batch 1 (indices 0-9)
    "employment_income",
    "long_term_capital_gains",
    "social_security",
    "taxable_pension_income",
    # Batch 2 (indices 10-19)
    "self_employment_income",
    # Batch 3 (indices 20-29)
    "rental_income",
    # Batch 4 (indices 30-39)
    "farm_income",
    # Batch 5 (indices 40-49)
    "traditional_ira_contributions",
    # Batch 6 (indices 50-59)
    "deductible_mortgage_interest",
    # Batch 7 (indices 60-66)
    "farm_operations_income",
]


class PeakMemoryMonitor:
    """Background thread sampling RSS at ~2 Hz to capture true peak."""

    def __init__(self):
        self._peak_bytes: int = 0
        self._running = False
        self._thread = None
        self._process = psutil.Process()

    def start(self) -> None:
        self._peak_bytes = self._process.memory_info().rss
        self._running = True
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def _sample(self) -> None:
        while self._running:
            rss = self._process.memory_info().rss
            if rss > self._peak_bytes:
                self._peak_bytes = rss
            time.sleep(0.5)

    def stop(self) -> int:
        """Stop monitoring and return peak RSS in bytes."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return self._peak_bytes


def compute_accuracy_metrics(
    imputed: Dict[str, np.ndarray],
    puf_reference: pd.DataFrame,
    puf_agi: np.ndarray,
) -> Dict[str, object]:
    """Compute distributional accuracy metrics for key variables.

    Args:
        imputed: Dict mapping variable name to imputed values.
        puf_reference: Full PUF training data (unsubsampled).
        puf_agi: Full PUF AGI array for top-tail analysis.

    Returns:
        Dict of metric name to value.
    """
    metrics: Dict[str, object] = {}
    agi_p995 = np.percentile(puf_agi, 99.5)

    for var in KEY_VARIABLES:
        if var not in imputed or var not in puf_reference.columns:
            continue

        imp_vals = imputed[var]
        ref_vals = puf_reference[var].values

        # KS statistic
        ks_stat, _ = stats.ks_2samp(imp_vals, ref_vals)
        metrics[f"{var}_ks"] = ks_stat

        # Wasserstein distance
        w_dist = stats.wasserstein_distance(imp_vals, ref_vals)
        metrics[f"{var}_wasserstein"] = w_dist

        # Marginal quantiles
        for q in [10, 25, 50, 75, 90, 95, 99]:
            metrics[f"{var}_p{q}"] = np.percentile(imp_vals, q)
        metrics[f"{var}_max"] = np.max(imp_vals)

        # Top-tail fidelity
        metrics[f"{var}_above_p995_agi"] = int(np.sum(imp_vals > agi_p995))
        metrics[f"{var}_imp_max"] = float(np.max(imp_vals))

    # Correlation preservation (Frobenius norm of diff)
    available = [
        v for v in KEY_VARIABLES if v in imputed and v in puf_reference
    ]
    if len(available) >= 2:
        imp_df = pd.DataFrame({v: imputed[v] for v in available})
        ref_df = puf_reference[available]
        imp_corr = imp_df.corr().values
        ref_corr = ref_df.corr().values
        frob_norm = np.linalg.norm(imp_corr - ref_corr, "fro")
        metrics["corr_frobenius_norm"] = frob_norm

    return metrics


def load_datasets(puf_path: str, cps_path: str) -> tuple:
    """Load PUF and CPS datasets via Microsimulation.

    Args:
        puf_path: Path to PUF dataset (h5 or class name).
        cps_path: Path to CPS dataset (h5 or class name).

    Returns:
        Tuple of (X_train_full, X_test, puf_agi,
                  puf_reference).
    """
    from policyengine_us import Microsimulation

    logger.info("Loading PUF dataset: %s", puf_path)
    puf_sim = Microsimulation(dataset=puf_path)
    puf_agi = puf_sim.calculate(
        "adjusted_gross_income", map_to="person"
    ).values
    X_train_full = puf_sim.calculate_dataframe(
        DEMOGRAPHIC_PREDICTORS + IMPUTED_VARIABLES
    )
    puf_reference = X_train_full[
        [v for v in KEY_VARIABLES if v in X_train_full.columns]
    ].copy()
    del puf_sim

    logger.info("Loading CPS dataset: %s", cps_path)
    cps_sim = Microsimulation(dataset=cps_path)
    X_test = cps_sim.calculate_dataframe(DEMOGRAPHIC_PREDICTORS)
    del cps_sim

    gc.collect()
    return X_train_full, X_test, puf_agi, puf_reference


def benchmark_single_size(
    size: int,
    X_train_full: pd.DataFrame,
    X_test: pd.DataFrame,
    puf_agi: np.ndarray,
    puf_reference: pd.DataFrame,
    monitor: PeakMemoryMonitor,
) -> Dict[str, object]:
    """Run benchmark for a single subsample size.

    Args:
        size: Target subsample size.
        X_train_full: Full PUF training data.
        X_test: CPS test data.
        puf_agi: Full PUF AGI array.
        puf_reference: Full PUF reference for accuracy.
        monitor: PeakMemoryMonitor instance.

    Returns:
        Dict of benchmark results for this size.
    """
    gc.collect()

    sub_idx = _stratified_subsample_index(puf_agi, target_n=size)
    actual_size = len(sub_idx)
    X_train_sub = X_train_full.iloc[sub_idx].reset_index(drop=True)

    logger.info("Benchmarking size=%d (actual=%d)", size, actual_size)

    # Start memory tracking
    monitor.start()
    tracemalloc.start()
    rss_before = psutil.Process().memory_info().rss

    t0 = time.perf_counter()
    y_full = _batch_qrf(
        X_train_sub,
        X_test,
        DEMOGRAPHIC_PREDICTORS,
        IMPUTED_VARIABLES,
    )
    wall_time = time.perf_counter() - t0

    _, peak_traced = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_rss = monitor.stop()
    rss_after = psutil.Process().memory_info().rss

    # Accuracy metrics
    accuracy = compute_accuracy_metrics(y_full, puf_reference, puf_agi)

    row = {
        "target_size": size,
        "actual_size": actual_size,
        "wall_time_s": round(wall_time, 1),
        "peak_traced_mb": round(peak_traced / 1024**2, 1),
        "peak_rss_mb": round(peak_rss / 1024**2, 1),
        "rss_delta_mb": round((rss_after - rss_before) / 1024**2, 1),
        "fits_gh_actions": peak_rss < GH_ACTIONS_MEMORY_LIMIT_BYTES,
    }
    row.update(accuracy)

    del y_full, X_train_sub
    gc.collect()

    return row


def run_benchmark(
    sizes: List[int],
    puf_path: str,
    cps_path: str,
    output_path: str,
) -> pd.DataFrame:
    """Run the full benchmark sweep.

    Args:
        sizes: List of subsample sizes to test.
        puf_path: Path to PUF dataset.
        cps_path: Path to CPS dataset.
        output_path: Path for CSV output.

    Returns:
        DataFrame with benchmark results.
    """
    X_train_full, X_test, puf_agi, puf_reference = load_datasets(
        puf_path, cps_path
    )

    logger.info(
        "PUF training records: %d, CPS test records: %d",
        len(X_train_full),
        len(X_test),
    )
    logger.info(
        "Imputing %d variables across %d sizes",
        len(IMPUTED_VARIABLES),
        len(sizes),
    )

    monitor = PeakMemoryMonitor()
    rows = []

    for size in sorted(sizes):
        if size > len(X_train_full):
            logger.warning(
                "Size %d exceeds PUF records (%d), skipping",
                size,
                len(X_train_full),
            )
            continue

        row = benchmark_single_size(
            size,
            X_train_full,
            X_test,
            puf_agi,
            puf_reference,
            monitor,
        )
        rows.append(row)

        logger.info(
            "  size=%d  wall=%.1fs  peak_rss=%.0fMB  " "gh_ok=%s",
            row["target_size"],
            row["wall_time_s"],
            row["peak_rss_mb"],
            row["fits_gh_actions"],
        )

    results = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results.to_csv(output_path, index=False)
    logger.info("Results saved to %s", output_path)

    return results


def print_summary(results: pd.DataFrame) -> None:
    """Print human-readable summary of benchmark results."""
    print("\n" + "=" * 70)
    print("QRF SUBSAMPLE BENCHMARK SUMMARY")
    print("=" * 70)

    print(
        f"\nMemory limit for GH Actions: "
        f"{GH_ACTIONS_MEMORY_LIMIT_GB:.0f} GB"
    )
    print(f"Variables imputed: {len(IMPUTED_VARIABLES)} " f"(full pass)")

    print(
        "\n{:<10} {:>8} {:>12} {:>12} {:>10}".format(
            "Size", "Time(s)", "Peak RSS(MB)", "Traced(MB)", "GH OK?"
        )
    )
    print("-" * 56)
    for _, r in results.iterrows():
        print(
            "{:<10} {:>8.1f} {:>12.0f} {:>12.0f} {:>10}".format(
                int(r["target_size"]),
                r["wall_time_s"],
                r["peak_rss_mb"],
                r["peak_traced_mb"],
                "YES" if r["fits_gh_actions"] else "NO",
            )
        )

    # Recommend max feasible size
    feasible = results[results["fits_gh_actions"]]
    if len(feasible) > 0:
        max_feasible = int(feasible["target_size"].max())
        print(f"\nRecommended max size for GH Actions: " f"{max_feasible:,}")
    else:
        print("\nNo tested size fits within GH Actions " "memory limit.")

    # Accuracy summary for key variables
    print("\nAccuracy (KS statistic, lower is better):")
    print("{:<10}".format("Size"), end="")
    for var in KEY_VARIABLES:
        col = f"{var}_ks"
        if col in results.columns:
            short = var.replace("_income", "").replace("_", " ")[:12]
            print(f" {short:>12}", end="")
    print()
    print("-" * (10 + 13 * len(KEY_VARIABLES)))
    for _, r in results.iterrows():
        print("{:<10}".format(int(r["target_size"])), end="")
        for var in KEY_VARIABLES:
            col = f"{var}_ks"
            if col in results.columns:
                print(f" {r[col]:>12.4f}", end="")
        print()

    if "corr_frobenius_norm" in results.columns:
        print("\nCorrelation Frobenius norm (lower=better):")
        for _, r in results.iterrows():
            print(
                f"  size={int(r['target_size']):>6}  "
                f"||dC||_F={r['corr_frobenius_norm']:.4f}"
            )

    print("\n" + "=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark QRF subsample sizes"
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[20_000, 40_000, 60_000, 80_000, 100_000],
        help="Subsample sizes to benchmark",
    )
    parser.add_argument(
        "--output",
        default="validation/outputs/subsample_benchmark.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--puf-dataset",
        required=True,
        help="Path to PUF dataset (h5 file or class)",
    )
    parser.add_argument(
        "--cps-dataset",
        required=True,
        help="Path to CPS dataset (h5 file or class)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: " "%(message)s",
    )

    logger.info("Starting benchmark with sizes: %s", args.sizes)

    results = run_benchmark(
        sizes=args.sizes,
        puf_path=args.puf_dataset,
        cps_path=args.cps_dataset,
        output_path=args.output,
    )

    print_summary(results)


if __name__ == "__main__":
    main()
