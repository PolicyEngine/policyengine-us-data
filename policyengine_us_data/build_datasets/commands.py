"""Command construction and execution for Stage 1 dataset builds."""

from __future__ import annotations

import subprocess
import sys
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any

from .results import DatasetCommandResult
from .status import Stage1ErrorRecord, utc_timestamp


@dataclass(frozen=True, kw_only=True)
class DatasetCommand:
    """A side-effecting command used by Stage 1 dataset builds."""

    name: str
    argv: tuple[str, ...]
    kind: str = "python"
    side_effecting: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_script(
        cls,
        script_path: str,
        *,
        args: Sequence[str] | None = None,
        python_executable: str | None = None,
    ) -> "DatasetCommand":
        """Build the command used to run a Python script or module."""

        script = Path(script_path)
        executable = python_executable or sys.executable
        if (
            script.suffix == ".py"
            and script.parts
            and script.parts[0] in {"policyengine_us_data", "modal_app"}
        ):
            argv = (
                executable,
                "-u",
                "-m",
                ".".join(script.with_suffix("").parts),
            )
        else:
            argv = (executable, "-u", script_path)
        if args:
            argv = (*argv, *tuple(args))
        return cls(
            name=script_path,
            argv=argv,
            kind="python_module" if "-m" in argv else "python_script",
            metadata={"script_path": script_path},
        )


class DatasetCommandError(RuntimeError):
    """Raised when a Stage 1 command exits unsuccessfully."""

    def __init__(self, result: DatasetCommandResult):
        self.result = result
        super().__init__(f"Command failed ({result.returncode}): {result.command_name}")


@dataclass
class SubprocessLogCapture:
    """Stream subprocess output while retaining a bounded output tail."""

    output_tail_lines: int = 200
    log_file: IO[str] | None = None
    _tail: deque[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._tail = deque(maxlen=max(self.output_tail_lines, 0))

    def write_line(self, line: str) -> None:
        """Write one subprocess line to stdout, the log file, and the tail."""

        sys.stdout.write(line)
        sys.stdout.flush()
        if self.log_file is not None:
            self.log_file.write(line)
        self._tail.append(line)

    def output_tail(self) -> tuple[str, ...]:
        """Return the retained subprocess output tail."""

        return tuple(self._tail)


@dataclass(frozen=True, kw_only=True)
class CommandRunner:
    """Run Stage 1 commands while streaming and capturing output."""

    output_tail_lines: int = 200

    def run(
        self,
        command: DatasetCommand,
        *,
        env: Mapping[str, str] | None = None,
        log_file: IO[str] | None = None,
        check: bool = True,
    ) -> DatasetCommandResult:
        """Run a command and return a structured execution result."""

        started_dt = datetime.now(timezone.utc)
        capture = SubprocessLogCapture(
            output_tail_lines=self.output_tail_lines,
            log_file=log_file,
        )
        run_env = dict(env) if env is not None else None
        if run_env is not None:
            run_env["PYTHONUNBUFFERED"] = "1"

        try:
            proc = subprocess.Popen(
                list(command.argv),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=run_env,
            )
            if proc.stdout is not None:
                for line in proc.stdout:
                    capture.write_line(line)
            proc.wait()
            result = self._result(
                command=command,
                started_dt=started_dt,
                returncode=proc.returncode,
                combined_output_tail=capture.output_tail(),
            )
            if check and proc.returncode != 0:
                raise DatasetCommandError(result)
            return result
        except DatasetCommandError:
            raise
        except Exception as exc:
            result = self._result(
                command=command,
                started_dt=started_dt,
                returncode=None,
                combined_output_tail=capture.output_tail(),
                exception=exc,
            )
            if check:
                raise DatasetCommandError(result) from exc
            return result

    def _result(
        self,
        *,
        command: DatasetCommand,
        started_dt: datetime,
        returncode: int | None,
        combined_output_tail: Sequence[str],
        exception: BaseException | None = None,
    ) -> DatasetCommandResult:
        completed_dt = datetime.now(timezone.utc)
        status = "completed" if returncode == 0 and exception is None else "failed"
        error = None
        if status == "failed":
            error = Stage1ErrorRecord.from_exception(
                exception or RuntimeError(f"Command exited with {returncode}"),
                command_name=command.name,
                returncode=returncode,
                metadata={
                    "argv": list(command.argv),
                    "kind": command.kind,
                    "output_tail": list(combined_output_tail),
                },
            )
        return DatasetCommandResult(
            command_name=command.name,
            argv=command.argv,
            status=status,
            returncode=returncode,
            started_at=utc_timestamp(started_dt),
            completed_at=utc_timestamp(completed_dt),
            duration_s=(completed_dt - started_dt).total_seconds(),
            combined_output_tail=tuple(combined_output_tail),
            error=error,
            metadata={
                **dict(command.metadata),
                "kind": command.kind,
                "side_effecting": command.side_effecting,
                "stderr_merged": True,
            },
        )


__all__ = [
    "CommandRunner",
    "DatasetCommand",
    "DatasetCommandError",
    "SubprocessLogCapture",
]
