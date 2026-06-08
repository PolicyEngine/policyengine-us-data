"""Source-level contracts for Makefile entrypoints."""

from __future__ import annotations

from pathlib import Path


MAKEFILE = Path("Makefile")


def _target_body(target: str) -> str:
    lines = MAKEFILE.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line == f"{target}:")
    body: list[str] = []
    for line in lines[start + 1 :]:
        if line and not line.startswith(("\t", " ", "#")) and line.endswith(":"):
            break
        body.append(line)
    return "\n".join(body)


def test_calibrate_modal_national_uses_unpenalized_national_default() -> None:
    body = _target_body("calibrate-modal-national")

    assert "--national" in body
    assert "--lambda-l0" not in body
