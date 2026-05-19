import os
import subprocess
import time


def create_secret_with_retry(args: list[str]) -> None:
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            subprocess.run(args, check=True)
            return
        except subprocess.CalledProcessError:
            if attempt == max_attempts:
                raise
            delay = min(2**attempt, 10)
            print(
                "Modal secret creation failed; retrying "
                f"in {delay}s ({attempt}/{max_attempts})"
            )
            time.sleep(delay)


def main() -> None:
    env_name = os.environ["MODAL_ENVIRONMENT"]
    create_secret_with_retry(
        [
            "modal",
            "secret",
            "create",
            "--env",
            env_name,
            "--force",
            "huggingface-token",
            f"HUGGING_FACE_TOKEN={os.environ['HUGGING_FACE_TOKEN']}",
        ],
    )
    create_secret_with_retry(
        [
            "modal",
            "secret",
            "create",
            "--env",
            env_name,
            "--force",
            "gcp-credentials",
            (
                "GOOGLE_APPLICATION_CREDENTIALS_JSON="
                f"{os.environ['GOOGLE_APPLICATION_CREDENTIALS']}"
            ),
        ],
    )


if __name__ == "__main__":
    main()
