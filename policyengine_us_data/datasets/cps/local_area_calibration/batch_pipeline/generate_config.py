#!/usr/bin/env python3
"""
Generate Cloud Batch job configuration from environment variables
"""
import json
import os
from pathlib import Path


def load_env_file(env_file=".env"):
    """Load environment variables from file"""
    if not Path(env_file).exists():
        env_file = "config.env"

    if Path(env_file).exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key] = value


def generate_config():
    """Generate batch_job_config.json from environment variables"""

    # Load environment variables
    load_env_file()

    # Parse allowed zones
    allowed_zones = os.getenv("ALLOWED_ZONES", "zones/us-central1-a").split(
        ","
    )

    config = {
        "taskGroups": [
            {
                "taskSpec": {
                    "runnables": [
                        {
                            "container": {
                                "imageUri": f"us-docker.pkg.dev/{os.getenv('PROJECT_ID')}/us.gcr.io/{os.getenv('IMAGE_NAME')}:{os.getenv('IMAGE_TAG', 'latest')}",
                                "entrypoint": "/app/run_batch_job.sh",
                            }
                        }
                    ],
                    "computeResource": {
                        "cpuMilli": int(os.getenv("CPU_MILLI", "8000")),
                        "memoryMib": int(os.getenv("MEMORY_MIB", "32768")),
                    },
                    "maxRunDuration": os.getenv("MAX_RUN_DURATION", "86400s"),
                    "environment": {
                        "variables": {
                            "BUCKET_NAME": os.getenv("BUCKET_NAME"),
                            "INPUT_PATH": os.getenv("INPUT_PATH"),
                            "OUTPUT_PATH": os.getenv("OUTPUT_PATH"),
                            "BETA": os.getenv("BETA", "0.35"),
                            "LAMBDA_L0": os.getenv("LAMBDA_L0", "5e-7"),
                            "LAMBDA_L2": os.getenv("LAMBDA_L2", "5e-9"),
                            "LR": os.getenv("LR", "0.1"),
                            "TOTAL_EPOCHS": os.getenv("TOTAL_EPOCHS", "12000"),
                            "EPOCHS_PER_CHUNK": os.getenv(
                                "EPOCHS_PER_CHUNK", "1000"
                            ),
                            "ENABLE_LOGGING": os.getenv(
                                "ENABLE_LOGGING", "true"
                            ),
                        }
                    },
                },
                "taskCount": 1,
                "parallelism": 1,
            }
        ],
        "allocationPolicy": {
            "instances": [
                {
                    "installGpuDrivers": True,
                    "policy": {
                        "machineType": os.getenv(
                            "MACHINE_TYPE", "n1-standard-2"
                        ),
                        "provisioningModel": os.getenv(
                            "PROVISIONING_MODEL", "SPOT"
                        ),
                        "accelerators": [
                            {
                                "type": os.getenv(
                                    "GPU_TYPE", "nvidia-tesla-p100"
                                ),
                                "count": int(os.getenv("GPU_COUNT", "1")),
                            }
                        ],
                        "bootDisk": {"sizeGb": "50"},
                    },
                }
            ],
            "location": {"allowedLocations": allowed_zones},
            "serviceAccount": {"email": os.getenv("SERVICE_ACCOUNT")},
        },
        "logsPolicy": {"destination": "CLOUD_LOGGING"},
    }

    # Write the configuration
    with open("batch_job_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("Generated batch_job_config.json from environment configuration")
    print(f"Project: {os.getenv('PROJECT_ID')}")
    print(
        f"Image: us-docker.pkg.dev/{os.getenv('PROJECT_ID')}/us.gcr.io/{os.getenv('IMAGE_NAME')}:{os.getenv('IMAGE_TAG')}"
    )
    print(f"GPU: {os.getenv('GPU_TYPE')}")
    print(f"Service Account: {os.getenv('SERVICE_ACCOUNT')}")


if __name__ == "__main__":
    generate_config()
