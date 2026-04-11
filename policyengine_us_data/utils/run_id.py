from datetime import datetime, timezone


def generate_run_id(version: str, sha: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{version}_{sha[:8]}_{ts}"
