import json
import os
import subprocess


def main() -> None:
    env_name = os.environ["MODAL_ENVIRONMENT"]
    environments = json.loads(
        subprocess.check_output(
            ["modal", "environment", "list", "--json"],
            text=True,
        )
    )
    if not any(item["name"] == env_name for item in environments):
        subprocess.run(
            ["modal", "environment", "create", env_name],
            check=True,
        )


if __name__ == "__main__":
    main()
