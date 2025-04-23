import os
import sys

# --- Check for Hugging Face Token ---
hf_token = os.getenv('HUGGING_FACE_TOKEN')

if not hf_token:
    warning_message = (
        "NOTE: HUGGING_FACE_TOKEN environment variable is not set.\n"
        "Functionality requiring authentication with the Hugging Face Hub may fail\n"
    )
    # Issue a warning (doesn't stop execution, but informs the user)
    # Print the message directly to standard error
    print(warning_message, file=sys.stderr)

# --- Continue with imports ---
from .datasets import *
from .geography import ZIP_CODE_DATASET
