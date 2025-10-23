#!/bin/bash
set -e

# Environment variables passed from Cloud Batch job config
BUCKET_NAME="${BUCKET_NAME:-policyengine-calibration}"
INPUT_PATH="${INPUT_PATH:-2024-10-08-2209/inputs}"
OUTPUT_PATH="${OUTPUT_PATH:-2024-10-08-2209/outputs}"

# Optimization parameters (can be overridden via env vars)
BETA="${BETA:-0.35}"
LAMBDA_L0="${LAMBDA_L0:-5e-7}"
LAMBDA_L2="${LAMBDA_L2:-5e-9}"
LR="${LR:-0.1}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-12000}"
EPOCHS_PER_CHUNK="${EPOCHS_PER_CHUNK:-1000}"
ENABLE_LOGGING="${ENABLE_LOGGING:-true}"

# Generate timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_ID="${JOB_ID:-batch_job_${TIMESTAMP}}"

echo "Starting Cloud Batch optimization job: ${JOB_ID}"
echo "Timestamp: ${TIMESTAMP}"
echo "Input: gs://${BUCKET_NAME}/${INPUT_PATH}"
echo "Output: gs://${BUCKET_NAME}/${OUTPUT_PATH}/${TIMESTAMP}"

# Create local working directories
LOCAL_INPUT="/tmp/input"
LOCAL_OUTPUT="/tmp/output"
mkdir -p ${LOCAL_INPUT}
mkdir -p ${LOCAL_OUTPUT}

# Download input data from GCS
echo "Downloading input data..."
gsutil cp "gs://${BUCKET_NAME}/${INPUT_PATH}/calibration_package.pkl" ${LOCAL_INPUT}/
gsutil cp "gs://${BUCKET_NAME}/${INPUT_PATH}/metadata.json" ${LOCAL_INPUT}/ 2>/dev/null || echo "No metadata.json found"

# Prepare logging flag
LOGGING_FLAG=""
if [ "${ENABLE_LOGGING}" = "true" ]; then
    LOGGING_FLAG="--enable-logging"
fi

# Run the optimization
echo "Starting optimization with parameters:"
echo "  Beta: ${BETA}"
echo "  Lambda L0: ${LAMBDA_L0}"
echo "  Lambda L2: ${LAMBDA_L2}"
echo "  Learning rate: ${LR}"
echo "  Total epochs: ${TOTAL_EPOCHS}"
echo "  Epochs per chunk: ${EPOCHS_PER_CHUNK}"
echo "  Device: cuda"

python /app/optimize_weights.py \
    --input-dir ${LOCAL_INPUT} \
    --output-dir ${LOCAL_OUTPUT} \
    --beta ${BETA} \
    --lambda-l0 ${LAMBDA_L0} \
    --lambda-l2 ${LAMBDA_L2} \
    --lr ${LR} \
    --total-epochs ${TOTAL_EPOCHS} \
    --epochs-per-chunk ${EPOCHS_PER_CHUNK} \
    ${LOGGING_FLAG} \
    --device cuda

# Upload results to GCS
echo "Uploading results to GCS..."
gsutil -m cp -r ${LOCAL_OUTPUT}/* "gs://${BUCKET_NAME}/${OUTPUT_PATH}/${TIMESTAMP}/"

# Create a completion marker
echo "{\"job_id\": \"${JOB_ID}\", \"timestamp\": \"${TIMESTAMP}\", \"status\": \"completed\"}" > ${LOCAL_OUTPUT}/job_complete.json
gsutil cp ${LOCAL_OUTPUT}/job_complete.json "gs://${BUCKET_NAME}/${OUTPUT_PATH}/${TIMESTAMP}/"

echo "Job completed successfully!"
echo "Results uploaded to: gs://${BUCKET_NAME}/${OUTPUT_PATH}/${TIMESTAMP}/"