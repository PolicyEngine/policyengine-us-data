"""Checkpoint adapter for Stage 1 dataset-build execution."""

from __future__ import annotations

import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


CheckpointAction = Literal["reuse", "recompute", "blocked"]


@dataclass(frozen=True, kw_only=True)
class CheckpointDecision:
    """Physical checkpoint decision for one expected output."""

    output_file: str
    checkpoint_path: Path
    action: CheckpointAction
    reason: str
    size_bytes: int = 0

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible checkpoint decision."""

        return {
            "output_file": self.output_file,
            "checkpoint_path": str(self.checkpoint_path),
            "action": self.action,
            "reason": self.reason,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, kw_only=True)
class CheckpointReuseSummary:
    """Checkpoint counter summary compatible with existing Stage 1 stats."""

    expected_outputs: int
    valid_reused_outputs: int
    recomputed_outputs: int
    invalid_outputs: int

    @classmethod
    def from_decisions(
        cls,
        decisions: Sequence[CheckpointDecision],
        *,
        recomputed: bool,
    ) -> "CheckpointReuseSummary":
        """Build prior-compatible counters from checkpoint decisions."""

        reusable = sum(decision.action == "reuse" for decision in decisions)
        invalid = sum(decision.action != "reuse" for decision in decisions)
        return cls(
            expected_outputs=len(decisions),
            valid_reused_outputs=reusable if not recomputed else 0,
            recomputed_outputs=len(decisions) if recomputed else 0,
            invalid_outputs=invalid,
        )

    def to_dict(self) -> dict[str, int]:
        """Return counters in the Stage 1 contract shape."""

        return {
            "expected_outputs": self.expected_outputs,
            "valid_reused_outputs": self.valid_reused_outputs,
            "recomputed_outputs": self.recomputed_outputs,
            "invalid_outputs": self.invalid_outputs,
        }


@dataclass(frozen=True, kw_only=True)
class CheckpointStore:
    """Adapter around the Stage 1 physical checkpoint volume layout."""

    root: Path
    branch: str
    commit_sha: str
    commit: Callable[[], None] | None = None

    def checkpoint_path(self, output_file: str) -> Path:
        """Return the checkpoint path for an output file."""

        return self.root / self.branch / self.commit_sha / Path(output_file).name

    def decision_for(self, output_file: str) -> CheckpointDecision:
        """Return the physical checkpoint decision for one output."""

        path = self.checkpoint_path(output_file)
        if not path.exists():
            return CheckpointDecision(
                output_file=output_file,
                checkpoint_path=path,
                action="recompute",
                reason="missing",
            )
        size = path.stat().st_size
        if size <= 0:
            return CheckpointDecision(
                output_file=output_file,
                checkpoint_path=path,
                action="recompute",
                reason="empty",
                size_bytes=size,
            )
        return CheckpointDecision(
            output_file=output_file,
            checkpoint_path=path,
            action="reuse",
            reason="valid",
            size_bytes=size,
        )

    def decisions_for(
        self, output_files: Sequence[str]
    ) -> tuple[CheckpointDecision, ...]:
        """Return physical checkpoint decisions for expected outputs."""

        return tuple(self.decision_for(output_file) for output_file in output_files)

    def all_outputs_reusable(self, output_files: Sequence[str]) -> bool:
        """Return true only when every expected output has a valid checkpoint."""

        return all(
            decision.action == "reuse" for decision in self.decisions_for(output_files)
        )

    def restore_output(self, output_file: str) -> bool:
        """Restore one checkpointed output if it is valid."""

        decision = self.decision_for(output_file)
        if decision.action != "reuse":
            return False
        local_path = Path(output_file)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(decision.checkpoint_path, local_path)
        return True

    def restore_all_outputs(self, output_files: Sequence[str]) -> bool:
        """Restore outputs only when all expected checkpoints are valid."""

        if not self.all_outputs_reusable(output_files):
            return False
        for output_file in output_files:
            self.restore_output(output_file)
        return True

    def save_output(self, output_file: str) -> bool:
        """Save one local output to the checkpoint store if it is non-empty."""

        local_path = Path(output_file)
        if not local_path.exists() or local_path.stat().st_size <= 0:
            return False
        checkpoint_path = self.checkpoint_path(output_file)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, checkpoint_path)
        if self.commit is not None:
            self.commit()
        return True

    def cleanup_branch(self) -> bool:
        """Remove all checkpoint attempts for this branch."""

        branch_dir = self.root / self.branch
        if not branch_dir.exists():
            return False
        shutil.rmtree(branch_dir)
        if self.commit is not None:
            self.commit()
        return True

    def cleanup_other_commits(self) -> tuple[Path, ...]:
        """Remove stale checkpoint directories for other commits in the branch."""

        branch_dir = self.root / self.branch
        if not branch_dir.exists():
            return ()
        removed: list[Path] = []
        for entry in branch_dir.iterdir():
            if entry.is_dir() and entry.name != self.commit_sha:
                shutil.rmtree(entry)
                removed.append(entry)
        if removed and self.commit is not None:
            self.commit()
        return tuple(removed)


__all__ = [
    "CheckpointDecision",
    "CheckpointReuseSummary",
    "CheckpointStore",
]
