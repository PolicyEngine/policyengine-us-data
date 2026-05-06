"""Deterministic JSON I/O helpers for stage contracts."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from policyengine_us_data.utils.canonical_json import (
    canonical_json_dumps,
    canonical_json_loads,
)

from .core import StageContract


def contract_to_json(contract: StageContract) -> str:
    """Serialize a stage contract to deterministic JSON."""

    return canonical_json_dumps(contract.to_dict())


def contract_from_json(payload: str) -> StageContract:
    """Deserialize deterministic JSON into a stage contract."""

    return StageContract.from_dict(canonical_json_loads(payload))


def write_contract(contract: StageContract, path: str | Path) -> None:
    """Write a stage contract to an explicit filesystem path."""

    contract_path = Path(path)
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with NamedTemporaryFile(
            "w",
            delete=False,
            dir=contract_path.parent,
            encoding="utf-8",
            prefix=f".{contract_path.name}.",
            suffix=".tmp",
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(contract_to_json(contract))
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.replace(contract_path)
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise


def read_contract(path: str | Path) -> StageContract:
    """Read a stage contract from an explicit filesystem path."""

    return contract_from_json(Path(path).read_text())
