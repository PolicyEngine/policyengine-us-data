from __future__ import annotations

import great_expectations as ge


def main() -> None:
    """Run Great Expectations validation on the policy data database."""
    # Ensure we load the DataContext from the repository root
    context = ge.get_context()
    # Execute the checkpoint configured for the policy database
    result = context.run_checkpoint(checkpoint_name="policy_data_checkpoint")
    if not result["success"]:
        raise ValueError("Great Expectations validation failed")
    print("Great Expectations validation succeeded")


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
