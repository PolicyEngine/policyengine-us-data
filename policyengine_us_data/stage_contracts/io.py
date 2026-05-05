"""Deterministic JSON I/O helpers for stage contracts."""

from __future__ import annotations

import json
from pathlib import Path

from .core import StageContract


def contract_to_json(contract: StageContract) -> str:
    """Serialize a stage contract to deterministic JSON."""

    return json.dumps(contract.to_dict(), indent=2, sort_keys=True) + "\n"


def contract_from_json(payload: str) -> StageContract:
    """Deserialize deterministic JSON into a stage contract."""

    return StageContract.from_dict(json.loads(payload))


def write_contract(contract: StageContract, path: str | Path) -> None:
    """Write a stage contract to an explicit filesystem path."""

    contract_path = Path(path)
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(contract_to_json(contract))


def read_contract(path: str | Path) -> StageContract:
    """Read a stage contract from an explicit filesystem path."""

    return contract_from_json(Path(path).read_text())
