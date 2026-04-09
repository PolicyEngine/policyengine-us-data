"""Re-upload files from Modal staging volume to HF staging."""

import sys
from pathlib import Path

_baked = "/root/policyengine-us-data"
_local = str(Path(__file__).resolve().parent)
for _p in (_baked, _local):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from modal_app.local_area import app, validate_staging, upload_to_staging

branch = "fix-would-file-blend-and-entity-weights"
version = "1.73.0"


@app.local_entrypoint()
def restage():
    print(f"Validating {version} on Modal volume...")
    manifest = validate_staging.remote(branch=branch, version=version)

    print(f"\nFound {len(manifest['files'])} files:")
    print(f"  States:    {manifest['totals']['states']}")
    print(f"  Districts: {manifest['totals']['districts']}")
    print(f"  Cities:    {manifest['totals']['cities']}")

    print("\nUploading to HF staging...")
    result = upload_to_staging.remote(branch=branch, version=version, manifest=manifest)
    print(result)
