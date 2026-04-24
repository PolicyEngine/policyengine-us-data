import argparse

from policyengine_us_data.utils.huggingface import upload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_path")
    parser.add_argument("repo_id")
    parser.add_argument("destination_path")
    args = parser.parse_args()
    upload(args.source_path, args.repo_id, args.destination_path)


if __name__ == "__main__":
    main()
