#!/bin/bash

# Script to build Docker image, push to GCR, and submit Cloud Batch job

# Load configuration from .env if it exists, otherwise use config.env
if [ -f .env ]; then
    echo "Loading configuration from .env"
    source .env
elif [ -f config.env ]; then
    echo "Loading configuration from config.env"
    source config.env
else
    echo "Error: No configuration file found. Please copy config.env to .env and customize it."
    exit 1
fi

# Allow command-line overrides
IMAGE_TAG="${1:-${IMAGE_TAG:-latest}}"
REGION="${2:-${REGION:-us-central1}}"
JOB_NAME="calibration-job-$(date +%Y%m%d-%H%M%S)"

echo "==========================================="
echo "Cloud Batch Calibration Job Submission"
echo "==========================================="
echo "Project: ${PROJECT_ID}"
echo "Image: us-docker.pkg.dev/${PROJECT_ID}/us.gcr.io/${IMAGE_NAME}:${IMAGE_TAG}"
echo "Region: ${REGION}"
echo "Job Name: ${JOB_NAME}"
echo ""

# Step 1: Build Docker image
echo "Step 1: Building Docker image..."
docker build -t us-docker.pkg.dev/${PROJECT_ID}/us.gcr.io/${IMAGE_NAME}:${IMAGE_TAG} .

if [ $? -ne 0 ]; then
    echo "Error: Docker build failed"
    exit 1
fi

# Step 2: Push to Artifact Registry
echo ""
echo "Step 2: Pushing image to Artifact Registry..."
docker push us-docker.pkg.dev/${PROJECT_ID}/us.gcr.io/${IMAGE_NAME}:${IMAGE_TAG}

if [ $? -ne 0 ]; then
    echo "Error: Docker push failed"
    echo "Make sure you're authenticated: gcloud auth configure-docker"
    exit 1
fi

# Step 3: Generate config and submit Cloud Batch job
echo ""
echo "Step 3a: Generating job configuration..."
python3 generate_config.py

echo ""
echo "Step 3b: Submitting Cloud Batch job..."
gcloud batch jobs submit ${JOB_NAME} \
    --location=${REGION} \
    --config=batch_job_config.json

if [ $? -eq 0 ]; then
    echo ""
    echo "==========================================="
    echo "Job submitted successfully!"
    echo "Job Name: ${JOB_NAME}"
    echo "Region: ${REGION}"
    echo ""
    echo "Monitor job status with:"
    echo "  gcloud batch jobs describe ${JOB_NAME} --location=${REGION}"
    echo ""
    echo "View logs with:"
    echo "  gcloud batch jobs list --location=${REGION}"
    echo "  gcloud logging read \"resource.type=batch.googleapis.com/Job AND resource.labels.job_id=${JOB_NAME}\" --limit=50"
    echo ""
    echo "Or use the monitoring script:"
    echo "  ./monitor_batch_job.sh ${JOB_NAME} ${REGION}"
    echo "==========================================="
else
    echo "Error: Job submission failed"
    exit 1
fi