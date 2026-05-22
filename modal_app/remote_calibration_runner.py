import os
import subprocess
import sys
from pathlib import Path

import modal

_baked = "/root/policyengine-us-data"
_local = str(Path(__file__).resolve().parent.parent)
for _p in (_baked, _local):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from modal_app.images import gpu_image as image  # noqa: E402
from policyengine_us_data.calibration_package.specs import (  # noqa: E402
    CALIBRATION_PACKAGE_CONTRACT_FILENAME,
    calibration_package_artifact_paths,
    stage2_build_context_for_run,
)
from policyengine_us_data.fit_weights import (  # noqa: E402
    FitResultBytes,
    FitScope,
    FittedWeightsInputBundle,
    NATIONAL_FIT_LAMBDA_L0,
    fit_artifacts_for_scope,
)

app = modal.App(
    os.environ.get("US_DATA_FIT_WEIGHTS_APP_NAME") or "policyengine-us-data-fit-weights"
)

hf_secret = modal.Secret.from_name("huggingface-token")
pipeline_vol = modal.Volume.from_name(
    os.environ.get("US_DATA_PIPELINE_VOLUME_NAME", "pipeline-artifacts"),
    create_if_missing=True,
    version=2,
)

PIPELINE_MOUNT = "/pipeline"


def _python_cmd(*args: str) -> list[str]:
    """Build a command that uses the current interpreter."""
    return [sys.executable, *args]


def _run_streaming(cmd, env=None, label=""):
    """Run a subprocess, streaming output line-by-line.

    Returns (returncode, captured_stdout_lines).
    """
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    lines = []
    for line in proc.stdout:
        line = line.rstrip("\n")
        if label:
            print(f"[{label}] {line}", flush=True)
        else:
            print(line, flush=True)
        lines.append(line)
    proc.wait()
    return proc.returncode, lines


def _setup_repo():
    """Change to the pre-baked repo directory."""
    os.chdir("/root/policyengine-us-data")


def _ensure_geography_prerequisites() -> None:
    """Download geography prerequisites excluded from the package wheel."""
    from policyengine_us_data.storage.download_prerequisites import (
        GEOGRAPHY_REPO,
        PREREQUISITE_ARTIFACTS,
        download_prerequisites,
    )

    geography_artifacts = tuple(
        artifact
        for artifact in PREREQUISITE_ARTIFACTS
        if artifact.repo == GEOGRAPHY_REPO
    )
    download_prerequisites(geography_artifacts)


def _append_hyperparams(cmd, beta, lambda_l0, lambda_l2, learning_rate, log_freq=None):
    """Append optional hyperparameter flags to a command list."""
    if beta is not None:
        cmd.extend(["--beta", str(beta)])
    if lambda_l0 is not None:
        cmd.extend(["--lambda-l0", str(lambda_l0)])
    if lambda_l2 is not None:
        cmd.extend(["--lambda-l2", str(lambda_l2)])
    if learning_rate is not None:
        cmd.extend(["--learning-rate", str(learning_rate)])
    if log_freq is not None:
        cmd.extend(["--log-freq", str(log_freq)])


def _collect_outputs(cal_lines):
    """Extract weights and log bytes from calibration output lines."""
    output_path = None
    geography_path = None
    log_path = None
    cal_log_path = None
    config_path = None
    for line in cal_lines:
        if "OUTPUT_PATH:" in line:
            output_path = line.split("OUTPUT_PATH:")[1].strip()
        elif "GEOGRAPHY_PATH:" in line:
            geography_path = line.split("GEOGRAPHY_PATH:")[1].strip()
        elif "CONFIG_PATH:" in line:
            config_path = line.split("CONFIG_PATH:")[1].strip()
        elif "CAL_LOG_PATH:" in line:
            cal_log_path = line.split("CAL_LOG_PATH:")[1].strip()
        elif "LOG_PATH:" in line:
            log_path = line.split("LOG_PATH:")[1].strip()

    with open(output_path, "rb") as f:
        weights_bytes = f.read()

    geography_bytes = None
    if geography_path:
        with open(geography_path, "rb") as f:
            geography_bytes = f.read()

    log_bytes = None
    if log_path:
        with open(log_path, "rb") as f:
            log_bytes = f.read()

    cal_log_bytes = None
    if cal_log_path:
        with open(cal_log_path, "rb") as f:
            cal_log_bytes = f.read()

    config_bytes = None
    if config_path:
        with open(config_path, "rb") as f:
            config_bytes = f.read()

    return FitResultBytes(
        weights=weights_bytes,
        geography=geography_bytes,
        diagnostics=log_bytes,
        epoch_log=cal_log_bytes,
        run_config=config_bytes,
    ).to_result_dict()


def _fit_output_filenames(
    *,
    scope: FitScope | str,
    output: str | None,
    log_output: str | None,
) -> dict[str, str]:
    """Return local and pipeline-volume filenames for one fit scope."""

    scope = FitScope.parse(scope)
    scoped = fit_artifacts_for_scope(scope)
    regional = fit_artifacts_for_scope(FitScope.REGIONAL)
    output = output or regional.weights.filename
    log_output = log_output or regional.diagnostics.filename
    if scope == FitScope.NATIONAL:
        output = (
            scoped.weights.filename
            if output == regional.weights.filename
            else f"national_{output}"
        )
        log_output = (
            scoped.diagnostics.filename
            if log_output == regional.diagnostics.filename
            else f"national_{log_output}"
        )
    return {
        "output": output,
        "log_output": log_output,
        "geography": scoped.geography.filename,
        "calibration_log": scoped.epoch_log.filename,
        "run_config": scoped.run_config.filename,
        "pipeline_weights": scoped.weights.filename,
        "pipeline_geography": scoped.geography.filename,
        "pipeline_run_config": scoped.run_config.filename,
    }


def _trigger_repository_dispatch(event_type: str = "calibration-updated"):
    """Fire a repository_dispatch event on GitHub."""
    import json
    import urllib.request

    token = os.environ.get(
        "GITHUB_TOKEN",
        os.environ.get("POLICYENGINE_US_DATA_GITHUB_TOKEN"),
    )
    if not token:
        print(
            "WARNING: No GITHUB_TOKEN or "
            "POLICYENGINE_US_DATA_GITHUB_TOKEN found. "
            "Skipping repository_dispatch.",
            flush=True,
        )
        return False

    url = "https://api.github.com/repos/PolicyEngine/policyengine-us-data/dispatches"
    payload = json.dumps({"event_type": event_type}).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    resp = urllib.request.urlopen(req)
    print(
        f"Triggered repository_dispatch '{event_type}' (HTTP {resp.status})",
        flush=True,
    )
    return True


def _fit_weights_impl(
    branch: str,
    epochs: int,
    target_config: str = None,
    beta: float = None,
    lambda_l0: float = None,
    lambda_l2: float = None,
    learning_rate: float = None,
    log_freq: int = None,
    skip_county: bool = True,
    workers: int = 8,
    artifacts_dir: str = "",
) -> dict:
    """Full pipeline: read data from pipeline volume, build matrix, fit."""
    _setup_repo()
    _ensure_geography_prerequisites()

    pipeline_vol.reload()
    artifacts = artifacts_dir if artifacts_dir else f"{PIPELINE_MOUNT}/artifacts"
    db_path = f"{artifacts}/policy_data.db"
    dataset_path = f"{artifacts}/source_imputed_stratified_extended_cps.h5"
    for label, p in [("database", db_path), ("dataset", dataset_path)]:
        if not os.path.exists(p):
            raise RuntimeError(
                f"Missing {label} on pipeline volume: {p}. Run data_build first."
            )

    cmd = [
        *_python_cmd("-m", "policyengine_us_data.calibration.unified_calibration"),
        "--device",
        "cuda",
        "--epochs",
        str(epochs),
        "--db-path",
        db_path,
        "--dataset",
        dataset_path,
    ]
    if target_config:
        cmd.extend(["--target-config", target_config])
    if not skip_county:
        cmd.append("--county-level")
    if workers > 1:
        cmd.extend(["--workers", str(workers)])
    _append_hyperparams(cmd, beta, lambda_l0, lambda_l2, learning_rate, log_freq)

    cal_rc, cal_lines = _run_streaming(
        cmd,
        env=os.environ.copy(),
        label="calibrate",
    )
    if cal_rc != 0:
        raise RuntimeError(f"Script failed with code {cal_rc}")

    return _collect_outputs(cal_lines)


def _fit_from_package_impl(
    branch: str,
    epochs: int,
    volume_package_path: str = None,
    volume_package_contract_path: str = None,
    allow_legacy_no_contract: bool = False,
    fit_scope: str = FitScope.REGIONAL.value,
    target_config: str = None,
    beta: float = None,
    lambda_l0: float = None,
    lambda_l2: float = None,
    learning_rate: float = None,
    log_freq: int = None,
) -> dict:
    """Fit weights from a pre-built calibration package."""
    if not volume_package_path:
        raise ValueError("volume_package_path is required")

    _setup_repo()
    input_bundle = FittedWeightsInputBundle(
        scope=fit_scope,
        calibration_package_path=Path(volume_package_path),
        calibration_package_contract_path=(
            Path(volume_package_contract_path) if volume_package_contract_path else None
        ),
        allow_legacy_no_contract=allow_legacy_no_contract,
    )
    stage2_identity = input_bundle.stage2_identity()
    if stage2_identity.stage2_contract_mode == "stage2_contract":
        print(
            "Validated Stage 2 calibration package contract "
            f"{stage2_identity.calibration_package_contract_fingerprint}",
            flush=True,
        )

    pkg_path = "/root/calibration_package.pkl"
    import shutil

    shutil.copy(volume_package_path, pkg_path)
    size = os.path.getsize(pkg_path)
    print(
        f"Copied package from volume ({size:,} bytes) to {pkg_path}",
        flush=True,
    )

    cmd = [
        *_python_cmd("-m", "policyengine_us_data.calibration.unified_calibration"),
        "--device",
        "cuda",
        "--epochs",
        str(epochs),
        "--package-path",
        pkg_path,
    ]
    if target_config:
        cmd.extend(["--target-config", target_config])
    _append_hyperparams(cmd, beta, lambda_l0, lambda_l2, learning_rate, log_freq)

    print(f"Running command: {' '.join(cmd)}", flush=True)

    cal_rc, cal_lines = _run_streaming(
        cmd,
        env=os.environ.copy(),
        label="calibrate",
    )
    if cal_rc != 0:
        raise RuntimeError(f"Script failed with code {cal_rc}")

    return _collect_outputs(cal_lines)


def _print_provenance_from_meta(meta: dict, current_branch: str = None) -> None:
    """Print provenance info and warn on branch mismatch."""
    built = meta.get("created_at", "unknown")
    branch = meta.get("git_branch", "unknown")
    commit = meta.get("git_commit")
    commit_short = commit[:8] if commit else "unknown"
    dirty = " (DIRTY)" if meta.get("git_dirty") else ""
    version = meta.get("package_version", "unknown")
    print("--- Package Provenance ---", flush=True)
    print(f"  Built:   {built}", flush=True)
    print(
        f"  Branch:  {branch} @ {commit_short}{dirty}",
        flush=True,
    )
    print(f"  Version: {version}", flush=True)
    print("--------------------------", flush=True)
    if current_branch and branch != "unknown" and branch != current_branch:
        print(
            f"WARNING: Package built on branch "
            f"'{branch}', but fitting with "
            f"--branch {current_branch}",
            flush=True,
        )


def _write_package_sidecar(pkg_path: str) -> bool:
    """Extract metadata from a pickle package and write a JSON sidecar.

    Returns:
        True if sidecar was written successfully, False otherwise.
    """
    import json
    import logging
    import pickle

    sidecar_path = pkg_path.replace(".pkl", "_meta.json")
    try:
        with open(pkg_path, "rb") as f:
            package = pickle.load(f)
        meta = package.get("metadata", {})
        del package
        with open(sidecar_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(
            f"Sidecar metadata written to {sidecar_path}",
            flush=True,
        )
        return True
    except Exception as e:
        logging.warning(
            "Failed to write package sidecar for %s: %s",
            pkg_path,
            e,
        )
        return False


def _build_package_impl(
    branch: str,
    target_config: str = None,
    skip_county: bool = True,
    workers: int = 8,
    n_clones: int = 430,
    run_id: str = "",
    modal_app_name: str = "",
    modal_environment: str = "",
    pipeline_volume_name: str = "",
    chunked_matrix: bool = False,
    chunk_size: int = 25_000,
    parallel_matrix: bool = False,
    num_matrix_workers: int = 50,
) -> str:
    """Read data from pipeline volume, build X matrix, save package."""
    _setup_repo()
    _ensure_geography_prerequisites()

    pipeline_vol.reload()
    build_context = stage2_build_context_for_run(
        PIPELINE_MOUNT, run_id
    ).require_inputs()
    input_bundle = build_context.input_bundle
    package_artifacts = build_context.output_bundle
    db_path = str(input_bundle.target_database)
    dataset_path = str(input_bundle.source_dataset)
    pkg_path = str(package_artifacts.package)
    cmd = [
        *_python_cmd("-m", "policyengine_us_data.calibration.unified_calibration"),
        "--device",
        "cpu",
        "--epochs",
        "0",
        "--db-path",
        db_path,
        "--dataset",
        dataset_path,
        "--build-only",
        "--package-output",
        pkg_path,
    ]
    if target_config:
        cmd.extend(["--target-config", target_config])
    if not skip_county:
        cmd.append("--county-level")
    if workers > 1:
        cmd.extend(["--workers", str(workers)])
    cmd.extend(["--n-clones", str(n_clones)])
    if chunked_matrix:
        cmd.extend(["--chunked-matrix", "--chunk-size", str(chunk_size)])
        if parallel_matrix:
            chunk_dir = str(package_artifacts.matrix_build_dir)
            cmd.extend(
                [
                    "--parallel",
                    "--chunk-dir",
                    chunk_dir,
                    "--num-matrix-workers",
                    str(num_matrix_workers),
                ]
            )

    build_env = os.environ.copy()
    if run_id:
        # ``unified_calibration.py`` reads this env var so workers can
        # locate their shared state at {pipeline-artifacts}/{run_id}/
        # matrix_build/chunk_build_state.pkl on the pipeline volume.
        build_env["POLICYENGINE_US_DATA_RUN_ID"] = run_id
        build_env["US_DATA_RUN_ID"] = run_id
    if modal_app_name:
        build_env["US_DATA_MODAL_APP_NAME"] = modal_app_name
        build_env["MODAL_APP_NAME"] = modal_app_name
    if modal_environment:
        build_env["US_DATA_MODAL_ENVIRONMENT"] = modal_environment
        build_env["MODAL_ENVIRONMENT"] = modal_environment
    if pipeline_volume_name:
        build_env["US_DATA_PIPELINE_VOLUME_NAME"] = pipeline_volume_name
    build_rc, build_lines = _run_streaming(
        cmd,
        env=build_env,
        label="build",
    )
    if build_rc != 0:
        raise RuntimeError(f"Package build failed with code {build_rc}")

    from policyengine_us_data.stage_contracts.calibration_package import (
        validate_persisted_calibration_package_contract,
    )

    validate_persisted_calibration_package_contract(
        package_path=package_artifacts.package,
        contract_path=package_artifacts.contract,
        dataset_path=Path(dataset_path),
        db_path=Path(db_path),
    )

    sidecar_ok = _write_package_sidecar(pkg_path)
    if not sidecar_ok:
        print(
            "WARNING: Package sidecar (provenance metadata) "
            "was not written. The package itself is still valid.",
            flush=True,
        )

    size = os.path.getsize(pkg_path)
    print(
        f"Package saved to volume at {pkg_path} ({size:,} bytes)",
        flush=True,
    )
    pipeline_vol.commit()
    return pkg_path


@app.function(
    image=image,
    secrets=[hf_secret],
    memory=65536,
    cpu=8.0,
    timeout=50400,
    volumes={PIPELINE_MOUNT: pipeline_vol},
    nonpreemptible=True,
)
def build_package_remote(
    branch: str = "main",
    target_config: str = None,
    skip_county: bool = True,
    workers: int = 8,
    n_clones: int = 430,
    run_id: str = "",
    modal_app_name: str = "",
    modal_environment: str = "",
    pipeline_volume_name: str = "",
    chunked_matrix: bool = False,
    chunk_size: int = 25_000,
    parallel_matrix: bool = False,
    num_matrix_workers: int = 50,
) -> str:
    return _build_package_impl(
        branch,
        target_config=target_config,
        skip_county=skip_county,
        workers=workers,
        n_clones=n_clones,
        run_id=run_id,
        modal_app_name=modal_app_name,
        modal_environment=modal_environment,
        pipeline_volume_name=pipeline_volume_name,
        chunked_matrix=chunked_matrix,
        chunk_size=chunk_size,
        parallel_matrix=parallel_matrix,
        num_matrix_workers=num_matrix_workers,
    )


@app.function(
    image=image,
    timeout=30,
    volumes={PIPELINE_MOUNT: pipeline_vol},
    nonpreemptible=True,
)
def check_volume_package(artifacts_dir: str = "") -> dict:
    """Check if a calibration package exists on the volume.

    Reads the lightweight JSON sidecar for provenance fields.
    Falls back to size/mtime if sidecar is missing.
    """
    import datetime
    import json

    base = artifacts_dir if artifacts_dir else f"{PIPELINE_MOUNT}/artifacts"
    package_artifacts = calibration_package_artifact_paths(base)
    pkg_path = str(package_artifacts.package)
    sidecar_path = str(package_artifacts.metadata)
    if not os.path.exists(pkg_path):
        return {"exists": False}

    stat = os.stat(pkg_path)
    mtime = datetime.datetime.fromtimestamp(stat.st_mtime, tz=datetime.timezone.utc)
    info = {
        "exists": True,
        "size": stat.st_size,
        "modified": mtime.strftime("%Y-%m-%d %H:%M UTC"),
    }
    if os.path.exists(sidecar_path):
        try:
            with open(sidecar_path) as f:
                meta = json.load(f)
            for key in (
                "git_branch",
                "git_commit",
                "git_dirty",
                "package_version",
                "created_at",
                "dataset_sha256",
                "db_sha256",
            ):
                if key in meta:
                    info[key] = meta[key]
        except Exception:
            pass
    return info


# --- Full pipeline GPU functions ---


@app.function(
    image=image,
    secrets=[hf_secret],
    memory=32768,
    cpu=8.0,
    gpu="T4",
    timeout=28800,
    volumes={PIPELINE_MOUNT: pipeline_vol},
)
def fit_weights_t4(
    branch: str = "main",
    epochs: int = 200,
    target_config: str = None,
    beta: float = None,
    lambda_l0: float = None,
    lambda_l2: float = None,
    learning_rate: float = None,
    log_freq: int = None,
    skip_county: bool = True,
    workers: int = 8,
    artifacts_dir: str = "",
) -> dict:
    return _fit_weights_impl(
        branch,
        epochs,
        target_config,
        beta,
        lambda_l0,
        lambda_l2,
        learning_rate,
        log_freq,
        skip_county=skip_county,
        workers=workers,
        artifacts_dir=artifacts_dir,
    )


@app.function(
    image=image,
    secrets=[hf_secret],
    memory=32768,
    cpu=8.0,
    gpu="A10",
    timeout=28800,
    volumes={PIPELINE_MOUNT: pipeline_vol},
)
def fit_weights_a10(
    branch: str = "main",
    epochs: int = 200,
    target_config: str = None,
    beta: float = None,
    lambda_l0: float = None,
    lambda_l2: float = None,
    learning_rate: float = None,
    log_freq: int = None,
    skip_county: bool = True,
    workers: int = 8,
    artifacts_dir: str = "",
) -> dict:
    return _fit_weights_impl(
        branch,
        epochs,
        target_config,
        beta,
        lambda_l0,
        lambda_l2,
        learning_rate,
        log_freq,
        skip_county=skip_county,
        workers=workers,
        artifacts_dir=artifacts_dir,
    )


@app.function(
    image=image,
    secrets=[hf_secret],
    memory=32768,
    cpu=8.0,
    gpu="A100-40GB",
    timeout=28800,
    volumes={PIPELINE_MOUNT: pipeline_vol},
)
def fit_weights_a100_40(
    branch: str = "main",
    epochs: int = 200,
    target_config: str = None,
    beta: float = None,
    lambda_l0: float = None,
    lambda_l2: float = None,
    learning_rate: float = None,
    log_freq: int = None,
    skip_county: bool = True,
    workers: int = 8,
    artifacts_dir: str = "",
) -> dict:
    return _fit_weights_impl(
        branch,
        epochs,
        target_config,
        beta,
        lambda_l0,
        lambda_l2,
        learning_rate,
        log_freq,
        skip_county=skip_county,
        workers=workers,
        artifacts_dir=artifacts_dir,
    )


@app.function(
    image=image,
    secrets=[hf_secret],
    memory=32768,
    cpu=8.0,
    gpu="A100-80GB",
    timeout=28800,
    volumes={PIPELINE_MOUNT: pipeline_vol},
)
def fit_weights_a100_80(
    branch: str = "main",
    epochs: int = 200,
    target_config: str = None,
    beta: float = None,
    lambda_l0: float = None,
    lambda_l2: float = None,
    learning_rate: float = None,
    log_freq: int = None,
    skip_county: bool = True,
    workers: int = 8,
    artifacts_dir: str = "",
) -> dict:
    return _fit_weights_impl(
        branch,
        epochs,
        target_config,
        beta,
        lambda_l0,
        lambda_l2,
        learning_rate,
        log_freq,
        skip_county=skip_county,
        workers=workers,
        artifacts_dir=artifacts_dir,
    )


@app.function(
    image=image,
    secrets=[hf_secret],
    memory=32768,
    cpu=8.0,
    gpu="H100",
    timeout=28800,
    volumes={PIPELINE_MOUNT: pipeline_vol},
)
def fit_weights_h100(
    branch: str = "main",
    epochs: int = 200,
    target_config: str = None,
    beta: float = None,
    lambda_l0: float = None,
    lambda_l2: float = None,
    learning_rate: float = None,
    log_freq: int = None,
    skip_county: bool = True,
    workers: int = 8,
    artifacts_dir: str = "",
) -> dict:
    return _fit_weights_impl(
        branch,
        epochs,
        target_config,
        beta,
        lambda_l0,
        lambda_l2,
        learning_rate,
        log_freq,
        skip_county=skip_county,
        workers=workers,
        artifacts_dir=artifacts_dir,
    )


GPU_FUNCTIONS = {
    "T4": fit_weights_t4,
    "A10": fit_weights_a10,
    "A100-40GB": fit_weights_a100_40,
    "A100-80GB": fit_weights_a100_80,
    "H100": fit_weights_h100,
}


# --- Package-path GPU functions ---


@app.function(
    image=image,
    memory=32768,
    cpu=8.0,
    gpu="T4",
    timeout=28800,
    volumes={PIPELINE_MOUNT: pipeline_vol},
)
def fit_from_package_t4(
    branch: str = "main",
    epochs: int = 200,
    target_config: str = None,
    beta: float = None,
    lambda_l0: float = None,
    lambda_l2: float = None,
    learning_rate: float = None,
    log_freq: int = None,
    volume_package_path: str = None,
    volume_package_contract_path: str = None,
    allow_legacy_no_contract: bool = False,
    fit_scope: str = FitScope.REGIONAL.value,
) -> dict:
    return _fit_from_package_impl(
        branch,
        epochs,
        volume_package_path=volume_package_path,
        volume_package_contract_path=volume_package_contract_path,
        allow_legacy_no_contract=allow_legacy_no_contract,
        fit_scope=fit_scope,
        target_config=target_config,
        beta=beta,
        lambda_l0=lambda_l0,
        lambda_l2=lambda_l2,
        learning_rate=learning_rate,
        log_freq=log_freq,
    )


@app.function(
    image=image,
    memory=32768,
    cpu=8.0,
    gpu="A10",
    timeout=28800,
    volumes={PIPELINE_MOUNT: pipeline_vol},
)
def fit_from_package_a10(
    branch: str = "main",
    epochs: int = 200,
    target_config: str = None,
    beta: float = None,
    lambda_l0: float = None,
    lambda_l2: float = None,
    learning_rate: float = None,
    log_freq: int = None,
    volume_package_path: str = None,
    volume_package_contract_path: str = None,
    allow_legacy_no_contract: bool = False,
    fit_scope: str = FitScope.REGIONAL.value,
) -> dict:
    return _fit_from_package_impl(
        branch,
        epochs,
        volume_package_path=volume_package_path,
        volume_package_contract_path=volume_package_contract_path,
        allow_legacy_no_contract=allow_legacy_no_contract,
        fit_scope=fit_scope,
        target_config=target_config,
        beta=beta,
        lambda_l0=lambda_l0,
        lambda_l2=lambda_l2,
        learning_rate=learning_rate,
        log_freq=log_freq,
    )


@app.function(
    image=image,
    memory=32768,
    cpu=8.0,
    gpu="A100-40GB",
    timeout=28800,
    volumes={PIPELINE_MOUNT: pipeline_vol},
)
def fit_from_package_a100_40(
    branch: str = "main",
    epochs: int = 200,
    target_config: str = None,
    beta: float = None,
    lambda_l0: float = None,
    lambda_l2: float = None,
    learning_rate: float = None,
    log_freq: int = None,
    volume_package_path: str = None,
    volume_package_contract_path: str = None,
    allow_legacy_no_contract: bool = False,
    fit_scope: str = FitScope.REGIONAL.value,
) -> dict:
    return _fit_from_package_impl(
        branch,
        epochs,
        volume_package_path=volume_package_path,
        volume_package_contract_path=volume_package_contract_path,
        allow_legacy_no_contract=allow_legacy_no_contract,
        fit_scope=fit_scope,
        target_config=target_config,
        beta=beta,
        lambda_l0=lambda_l0,
        lambda_l2=lambda_l2,
        learning_rate=learning_rate,
        log_freq=log_freq,
    )


@app.function(
    image=image,
    memory=32768,
    cpu=8.0,
    gpu="A100-80GB",
    timeout=28800,
    volumes={PIPELINE_MOUNT: pipeline_vol},
)
def fit_from_package_a100_80(
    branch: str = "main",
    epochs: int = 200,
    target_config: str = None,
    beta: float = None,
    lambda_l0: float = None,
    lambda_l2: float = None,
    learning_rate: float = None,
    log_freq: int = None,
    volume_package_path: str = None,
    volume_package_contract_path: str = None,
    allow_legacy_no_contract: bool = False,
    fit_scope: str = FitScope.REGIONAL.value,
) -> dict:
    return _fit_from_package_impl(
        branch,
        epochs,
        volume_package_path=volume_package_path,
        volume_package_contract_path=volume_package_contract_path,
        allow_legacy_no_contract=allow_legacy_no_contract,
        fit_scope=fit_scope,
        target_config=target_config,
        beta=beta,
        lambda_l0=lambda_l0,
        lambda_l2=lambda_l2,
        learning_rate=learning_rate,
        log_freq=log_freq,
    )


@app.function(
    image=image,
    memory=32768,
    cpu=8.0,
    gpu="H100",
    timeout=28800,
    volumes={PIPELINE_MOUNT: pipeline_vol},
)
def fit_from_package_h100(
    branch: str = "main",
    epochs: int = 200,
    target_config: str = None,
    beta: float = None,
    lambda_l0: float = None,
    lambda_l2: float = None,
    learning_rate: float = None,
    log_freq: int = None,
    volume_package_path: str = None,
    volume_package_contract_path: str = None,
    allow_legacy_no_contract: bool = False,
    fit_scope: str = FitScope.REGIONAL.value,
) -> dict:
    return _fit_from_package_impl(
        branch,
        epochs,
        volume_package_path=volume_package_path,
        volume_package_contract_path=volume_package_contract_path,
        allow_legacy_no_contract=allow_legacy_no_contract,
        fit_scope=fit_scope,
        target_config=target_config,
        beta=beta,
        lambda_l0=lambda_l0,
        lambda_l2=lambda_l2,
        learning_rate=learning_rate,
        log_freq=log_freq,
    )


PACKAGE_GPU_FUNCTIONS = {
    "T4": fit_from_package_t4,
    "A10": fit_from_package_a10,
    "A100-40GB": fit_from_package_a100_40,
    "A100-80GB": fit_from_package_a100_80,
    "H100": fit_from_package_h100,
}


@app.local_entrypoint()
def main(
    branch: str = "main",
    epochs: int = 200,
    gpu: str = "T4",
    output: str = "calibration_weights.npy",
    log_output: str = "unified_diagnostics.csv",
    target_config: str = None,
    beta: float = None,
    lambda_l0: float = None,
    lambda_l2: float = None,
    learning_rate: float = None,
    log_freq: int = None,
    package_path: str = None,
    full_pipeline: bool = False,
    county_level: bool = False,
    workers: int = 8,
    push_results: bool = False,
    trigger_publish: bool = False,
    national: bool = False,
):
    scope = FitScope.NATIONAL if national else FitScope.REGIONAL
    prefix = "national_" if national else ""
    output_filenames = _fit_output_filenames(
        scope=scope,
        output=output,
        log_output=log_output,
    )
    output = output_filenames["output"]
    log_output = output_filenames["log_output"]
    if national:
        if lambda_l0 is None:
            lambda_l0 = NATIONAL_FIT_LAMBDA_L0

    if gpu not in GPU_FUNCTIONS:
        raise ValueError(
            f"Unknown GPU: {gpu}. Choose from: {list(GPU_FUNCTIONS.keys())}"
        )

    if package_path:
        vol_path = f"{PIPELINE_MOUNT}/artifacts/calibration_package.pkl"
        local_contract_path = Path(package_path).with_name(
            CALIBRATION_PACKAGE_CONTRACT_FILENAME
        )
        vol_contract_path = (
            f"{PIPELINE_MOUNT}/artifacts/{CALIBRATION_PACKAGE_CONTRACT_FILENAME}"
            if local_contract_path.exists()
            else None
        )
        print(f"Reading package from {package_path}...", flush=True)
        import json as _json
        import pickle as _pkl

        with open(package_path, "rb") as f:
            package_bytes = f.read()
        contract_bytes = (
            local_contract_path.read_bytes() if local_contract_path.exists() else None
        )
        size = len(package_bytes)
        pkg_meta = _pkl.loads(package_bytes).get("metadata", {})
        sidecar_bytes = _json.dumps(pkg_meta, indent=2).encode()
        print(
            f"Uploading package ({size:,} bytes) to Modal volume...",
            flush=True,
        )
        with pipeline_vol.batch_upload(force=True) as batch:
            from io import BytesIO

            batch.put_file(
                BytesIO(package_bytes),
                "artifacts/calibration_package.pkl",
            )
            batch.put_file(
                BytesIO(sidecar_bytes),
                "artifacts/calibration_package_meta.json",
            )
            if contract_bytes is not None:
                batch.put_file(
                    BytesIO(contract_bytes),
                    f"artifacts/{CALIBRATION_PACKAGE_CONTRACT_FILENAME}",
                )
        pipeline_vol.commit()
        del package_bytes
        print("Upload complete.", flush=True)
        _print_provenance_from_meta(pkg_meta, branch)
        func = PACKAGE_GPU_FUNCTIONS[gpu]
        result = func.remote(
            branch=branch,
            epochs=epochs,
            target_config=target_config,
            beta=beta,
            lambda_l0=lambda_l0,
            lambda_l2=lambda_l2,
            learning_rate=learning_rate,
            log_freq=log_freq,
            volume_package_path=vol_path,
            volume_package_contract_path=vol_contract_path,
            allow_legacy_no_contract=True,
            fit_scope=scope.value,
        )
    elif full_pipeline:
        print(
            "========================================",
            flush=True,
        )
        print(
            "Mode: full pipeline (read from volume, build matrix, fit)",
            flush=True,
        )
        print(
            f"GPU: {gpu} | Epochs: {epochs} | Branch: {branch}",
            flush=True,
        )
        print(
            "========================================",
            flush=True,
        )
        func = GPU_FUNCTIONS[gpu]
        result = func.remote(
            branch=branch,
            epochs=epochs,
            target_config=target_config,
            beta=beta,
            lambda_l0=lambda_l0,
            lambda_l2=lambda_l2,
            learning_rate=learning_rate,
            log_freq=log_freq,
            skip_county=not county_level,
            workers=workers,
        )
    else:
        vol_path = f"{PIPELINE_MOUNT}/artifacts/calibration_package.pkl"
        vol_contract_path = (
            f"{PIPELINE_MOUNT}/artifacts/{CALIBRATION_PACKAGE_CONTRACT_FILENAME}"
        )
        vol_info = check_volume_package.remote()
        if not vol_info["exists"]:
            raise SystemExit(
                "\nNo calibration package found on Modal volume.\n"
                "Run 'make build-matrices' first, or use "
                "--full-pipeline to build from scratch.\n"
            )
        if vol_info.get("created_at") or vol_info.get("git_branch"):
            _print_provenance_from_meta(vol_info, branch)
        mode_label = (
            "national calibration" if national else "fitting from pre-built package"
        )
        print(
            "========================================",
            flush=True,
        )
        print(f"Mode: {mode_label}", flush=True)
        print(
            f"GPU: {gpu} | Epochs: {epochs} | Branch: {branch}",
            flush=True,
        )
        if push_results:
            print(
                "After fitting, will upload to HuggingFace:",
                flush=True,
            )
            print(
                f"  - calibration/{output_filenames['pipeline_weights']}",
                flush=True,
            )
            print(
                f"  - calibration/{output_filenames['pipeline_geography']}",
                flush=True,
            )
            print(
                f"  - calibration/logs/{prefix}* (diagnostics, "
                "config, calibration log)",
                flush=True,
            )
        print(
            "========================================",
            flush=True,
        )
        func = PACKAGE_GPU_FUNCTIONS[gpu]
        result = func.remote(
            branch=branch,
            epochs=epochs,
            target_config=target_config,
            beta=beta,
            lambda_l0=lambda_l0,
            lambda_l2=lambda_l2,
            learning_rate=learning_rate,
            log_freq=log_freq,
            volume_package_path=vol_path,
            volume_package_contract_path=vol_contract_path,
            allow_legacy_no_contract=True,
            fit_scope=scope.value,
        )

    with open(output, "wb") as f:
        f.write(result["weights"])
    print(f"Weights saved to: {output}")

    if result["log"]:
        with open(log_output, "wb") as f:
            f.write(result["log"])
        print(f"Diagnostics log saved to: {log_output}")

    geography_output = output_filenames["geography"]
    if result.get("geography"):
        with open(geography_output, "wb") as f:
            f.write(result["geography"])
        print(f"Geography saved to: {geography_output}")

    cal_log_output = output_filenames["calibration_log"]
    if result.get("cal_log"):
        with open(cal_log_output, "wb") as f:
            f.write(result["cal_log"])
        print(f"Calibration log saved to: {cal_log_output}")

    config_output = output_filenames["run_config"]
    if result.get("config"):
        with open(config_output, "wb") as f:
            f.write(result["config"])
        print(f"Run config saved to: {config_output}")

    # Push weights to pipeline volume for downstream steps
    from io import BytesIO

    print("Pushing weights to pipeline volume...", flush=True)
    with pipeline_vol.batch_upload(force=True) as batch:
        batch.put_file(
            BytesIO(result["weights"]),
            f"artifacts/{output_filenames['pipeline_weights']}",
        )
        if result.get("geography"):
            batch.put_file(
                BytesIO(result["geography"]),
                f"artifacts/{output_filenames['pipeline_geography']}",
            )
        if result.get("config"):
            batch.put_file(
                BytesIO(result["config"]),
                f"artifacts/{output_filenames['pipeline_run_config']}",
            )
    pipeline_vol.commit()
    print("Weights committed to pipeline volume", flush=True)

    if push_results:
        from policyengine_us_data.utils.huggingface import (
            upload_calibration_artifacts,
        )

        upload_calibration_artifacts(
            weights_path=output,
            geography_path=(geography_output if result.get("geography") else None),
            log_dir=".",
            prefix=prefix,
        )

    if trigger_publish:
        _trigger_repository_dispatch()


@app.local_entrypoint()
def build_package(
    branch: str = "main",
    target_config: str = None,
    county_level: bool = False,
    workers: int = 8,
    n_clones: int = 430,
):
    """Build the calibration package (X matrix) on CPU and save
    to Modal volume. Then run main() to fit."""
    print(
        "========================================",
        flush=True,
    )
    print("Mode: building calibration package (CPU only)", flush=True)
    print(f"Branch: {branch}", flush=True)
    print(
        "This builds the X matrix and saves it to a Modal volume.",
        flush=True,
    )
    print(
        "No GPU is used. Timeout: 14 hours.",
        flush=True,
    )
    print(
        "========================================",
        flush=True,
    )
    vol_path = build_package_remote.remote(
        branch=branch,
        target_config=target_config,
        skip_county=not county_level,
        workers=workers,
        n_clones=n_clones,
    )
    print(
        f"Package built and saved to Modal volume at {vol_path}",
        flush=True,
    )
    print(
        "\nTo fit weights, run:\n"
        "  modal run modal_app/remote_calibration_runner.py"
        "::main \\\n"
        f"    --branch {branch} --gpu <GPU> "
        "--epochs <N> --push-results",
        flush=True,
    )
