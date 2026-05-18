import numpy as np


def _validate_same_shape(*arrays: np.ndarray) -> None:
    shapes = {np.asarray(array).shape for array in arrays}
    if len(shapes) != 1:
        raise ValueError("All arrays must have the same shape")


def prioritize_reported_recipients(
    reported_receipt: np.ndarray,
    target_rate: float,
    draws: np.ndarray,
    eligible_mask: np.ndarray | None = None,
) -> np.ndarray:
    reported_receipt = np.asarray(reported_receipt, dtype=bool)
    draws = np.asarray(draws)
    _validate_same_shape(reported_receipt, draws)
    if eligible_mask is None:
        eligible_mask = np.ones(reported_receipt.shape, dtype=bool)
    else:
        eligible_mask = np.asarray(eligible_mask, dtype=bool)
        _validate_same_shape(reported_receipt, eligible_mask)

    eligible_mask = eligible_mask | reported_receipt
    n_reporters = reported_receipt.sum()
    n_non_reporters = (eligible_mask & ~reported_receipt).sum()
    n_entities = eligible_mask.sum()
    target_takeup_count = int(target_rate * n_entities)
    remaining_needed = max(0, target_takeup_count - n_reporters)
    non_reporter_rate = remaining_needed / n_non_reporters if n_non_reporters > 0 else 0

    return reported_receipt | (
        eligible_mask & (~reported_receipt) & (draws < non_reporter_rate)
    )


def align_reported_ssi_disability(
    is_disabled: np.ndarray, reported_ssi: np.ndarray, ages: np.ndarray
) -> np.ndarray:
    is_disabled = np.asarray(is_disabled, dtype=bool)
    reported_ssi = np.asarray(reported_ssi, dtype=bool)
    ages = np.asarray(ages)
    _validate_same_shape(is_disabled, reported_ssi, ages)
    return is_disabled | (reported_ssi & (ages < 65))
