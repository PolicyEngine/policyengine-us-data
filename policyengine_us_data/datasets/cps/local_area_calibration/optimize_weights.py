#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
from datetime import datetime
import pickle
import torch
import numpy as np
from scipy import sparse as sp
from l0.calibration import SparseCalibrationWeights


def main():
    parser = argparse.ArgumentParser(
        description="Run sparse L0 weight optimization"
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing calibration_package.pkl",
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory for output files"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.35,
        help="Beta parameter for L0 regularization",
    )
    parser.add_argument(
        "--lambda-l0",
        type=float,
        default=5e-7,
        help="L0 regularization strength",
    )
    parser.add_argument(
        "--lambda-l2",
        type=float,
        default=5e-9,
        help="L2 regularization strength",
    )
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument(
        "--total-epochs", type=int, default=12000, help="Total training epochs"
    )
    parser.add_argument(
        "--epochs-per-chunk",
        type=int,
        default=1000,
        help="Epochs per logging chunk",
    )
    parser.add_argument(
        "--enable-logging",
        action="store_true",
        help="Enable detailed epoch logging",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use",
    )

    args = parser.parse_args()

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading calibration package from {args.input_dir}")
    with open(Path(args.input_dir) / "calibration_package.pkl", "rb") as f:
        calibration_data = pickle.load(f)

    X_sparse = calibration_data["X_sparse"]
    init_weights = calibration_data["initial_weights"]
    targets_df = calibration_data["targets_df"]
    targets = targets_df.value.values

    print(f"Matrix shape: {X_sparse.shape}")
    print(f"Number of targets: {len(targets)}")

    target_names = []
    for _, row in targets_df.iterrows():
        geo_prefix = f"{row['geographic_id']}"
        name = f"{geo_prefix}/{row['variable_desc']}"
        target_names.append(name)

    model = SparseCalibrationWeights(
        n_features=X_sparse.shape[1],
        beta=args.beta,
        gamma=-0.1,
        zeta=1.1,
        init_keep_prob=0.999,
        init_weights=init_weights,
        log_weight_jitter_sd=0.05,
        log_alpha_jitter_sd=0.01,
        device=args.device,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.enable_logging:
        log_path = output_dir / "cd_calibration_log.csv"
        with open(log_path, "w") as f:
            f.write(
                "target_name,estimate,target,epoch,error,rel_error,abs_error,rel_abs_error,loss\n"
            )
        print(f"Initialized incremental log at: {log_path}")

    sparsity_path = output_dir / f"cd_sparsity_history_{timestamp}.csv"
    with open(sparsity_path, "w") as f:
        f.write("epoch,active_weights,total_weights,sparsity_pct\n")
    print(f"Initialized sparsity tracking at: {sparsity_path}")

    for chunk_start in range(0, args.total_epochs, args.epochs_per_chunk):
        chunk_epochs = min(
            args.epochs_per_chunk, args.total_epochs - chunk_start
        )
        current_epoch = chunk_start + chunk_epochs

        print(
            f"\nTraining epochs {chunk_start + 1} to {current_epoch} of {args.total_epochs}"
        )

        model.fit(
            M=X_sparse,
            y=targets,
            target_groups=None,
            lambda_l0=args.lambda_l0,
            lambda_l2=args.lambda_l2,
            lr=args.lr,
            epochs=chunk_epochs,
            loss_type="relative",
            verbose=True,
            verbose_freq=chunk_epochs,
        )

        active_info = model.get_active_weights()
        active_count = active_info["count"]
        total_count = X_sparse.shape[1]
        sparsity_pct = 100 * (1 - active_count / total_count)

        with open(sparsity_path, "a") as f:
            f.write(
                f"{current_epoch},{active_count},{total_count},{sparsity_pct:.4f}\n"
            )

        if args.enable_logging:
            with torch.no_grad():
                y_pred = model.predict(X_sparse).cpu().numpy()

                with open(log_path, "a") as f:
                    for i in range(len(targets)):
                        estimate = y_pred[i]
                        target = targets[i]
                        error = estimate - target
                        rel_error = error / target if target != 0 else 0
                        abs_error = abs(error)
                        rel_abs_error = abs(rel_error)
                        loss = rel_error**2

                        f.write(
                            f'"{target_names[i]}",{estimate},{target},{current_epoch},'
                            f"{error},{rel_error},{abs_error},{rel_abs_error},{loss}\n"
                        )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    with torch.no_grad():
        w = model.get_weights(deterministic=True).cpu().numpy()

    versioned_filename = f"w_cd_{timestamp}.npy"
    full_path = output_dir / versioned_filename
    np.save(full_path, w)

    canonical_path = output_dir / "w_cd.npy"
    np.save(canonical_path, w)

    print(f"\nOptimization complete!")
    print(f"Final weights saved to: {full_path}")
    print(f"Canonical weights saved to: {canonical_path}")
    print(f"Weights shape: {w.shape}")
    print(f"Sparsity history saved to: {sparsity_path}")


if __name__ == "__main__":
    main()
