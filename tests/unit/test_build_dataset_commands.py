import sys

import pytest

from policyengine_us_data.build_datasets import (
    CommandRunner,
    DatasetCommand,
    DatasetCommandError,
)


def test_dataset_command_builds_python_module_command():
    command = DatasetCommand.from_script(
        "policyengine_us_data/datasets/cps/cps.py",
        python_executable="/python",
    )

    assert command.argv == (
        "/python",
        "-u",
        "-m",
        "policyengine_us_data.datasets.cps.cps",
    )
    assert command.name == "policyengine_us_data/datasets/cps/cps.py"
    assert command.metadata["script_path"] == command.name


def test_dataset_command_keeps_external_python_script_path():
    command = DatasetCommand.from_script(
        "scripts/example.py",
        args=("--flag",),
        python_executable="/python",
    )

    assert command.argv == ("/python", "-u", "scripts/example.py", "--flag")
    assert command.kind == "python_script"


def test_dataset_command_represents_side_effecting_make_command():
    command = DatasetCommand(
        name="make database",
        argv=("make", "database"),
        kind="side_effect",
    )

    assert command.side_effecting is True
    assert command.argv == ("make", "database")


def test_command_runner_raises_structured_failure():
    command = DatasetCommand(
        name="failing command",
        argv=(
            sys.executable,
            "-c",
            "import sys; print('structured failure'); sys.exit(7)",
        ),
    )

    with pytest.raises(DatasetCommandError) as exc_info:
        CommandRunner(output_tail_lines=5).run(command)

    result = exc_info.value.result
    assert result.status == "failed"
    assert result.returncode == 7
    assert result.error is not None
    assert result.error.command_name == "failing command"
    assert result.combined_output_tail == ("structured failure\n",)


def test_command_runner_can_return_structured_failure_without_raising():
    command = DatasetCommand(
        name="nonraising command",
        argv=(sys.executable, "-c", "import sys; sys.exit(3)"),
    )

    result = CommandRunner().run(command, check=False)

    assert result.status == "failed"
    assert result.returncode == 3
    assert result.error is not None
    assert result.error.returncode == 3
