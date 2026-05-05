import os
import subprocess


def main() -> None:
    env_name = os.environ["MODAL_ENVIRONMENT"]
    subprocess.run(
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
        check=True,
    )
    subprocess.run(
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
        check=True,
    )


if __name__ == "__main__":
    main()
