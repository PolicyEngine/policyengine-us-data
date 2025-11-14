#!/bin/bash

# Monitor Cloud Batch job status

JOB_NAME="${1}"
REGION="${2:-us-central1}"

if [ -z "${JOB_NAME}" ]; then
    echo "Usage: $0 <job_name> [region]"
    echo "Example: $0 calibration-job-20241015-143022 us-central1"
    exit 1
fi

echo "Monitoring job: ${JOB_NAME}"
echo "Region: ${REGION}"
echo "Press Ctrl+C to stop monitoring"
echo ""

# Function to get job status
get_status() {
    gcloud batch jobs describe ${JOB_NAME} \
        --location=${REGION} \
        --format="value(status.state)" 2>/dev/null
}

# Monitor loop
while true; do
    STATUS=$(get_status)
    TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")

    case ${STATUS} in
        "SUCCEEDED")
            echo "[${TIMESTAMP}] Job ${JOB_NAME} completed successfully!"
            echo ""
            echo "Fetching final logs..."
            gcloud logging read "resource.type=batch.googleapis.com/Job AND resource.labels.job_id=${JOB_NAME}" \
                --limit=100 \
                --format="table(timestamp,severity,textPayload)"
            echo ""
            echo "Job completed! Check your GCS bucket for results."
            exit 0
            ;;
        "FAILED")
            echo "[${TIMESTAMP}] Job ${JOB_NAME} failed!"
            echo ""
            echo "Fetching error logs..."
            gcloud logging read "resource.type=batch.googleapis.com/Job AND resource.labels.job_id=${JOB_NAME} AND severity>=ERROR" \
                --limit=50 \
                --format="table(timestamp,severity,textPayload)"
            exit 1
            ;;
        "RUNNING")
            echo "[${TIMESTAMP}] Job is running..."
            # Optionally fetch recent logs
            echo "Recent logs:"
            gcloud logging read "resource.type=batch.googleapis.com/Job AND resource.labels.job_id=${JOB_NAME}" \
                --limit=5 \
                --format="table(timestamp,textPayload)" 2>/dev/null
            ;;
        "PENDING"|"QUEUED"|"SCHEDULED")
            echo "[${TIMESTAMP}] Job status: ${STATUS} - waiting for resources..."
            ;;
        *)
            echo "[${TIMESTAMP}] Job status: ${STATUS}"
            ;;
    esac

    sleep 30
done