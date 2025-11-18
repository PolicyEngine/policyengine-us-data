#!/bin/bash

echo "========================================="
echo "Cloud Batch Pipeline Setup"
echo "========================================="
echo ""

# Check prerequisites
echo "Checking prerequisites..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed"
    echo "   Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
else
    echo "✅ Docker is installed: $(docker --version)"
fi

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI is not installed"
    echo "   Please install gcloud: https://cloud.google.com/sdk/docs/install"
    exit 1
else
    echo "✅ gcloud is installed: $(gcloud --version | head -n 1)"
fi

# Check if authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" &> /dev/null; then
    echo "❌ Not authenticated with gcloud"
    echo "   Please run: gcloud auth login"
    exit 1
else
    ACTIVE_ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format="value(account)")
    echo "✅ Authenticated as: ${ACTIVE_ACCOUNT}"
fi

# Check Docker authentication for GCR
echo ""
echo "Configuring Docker for Google Container Registry..."
gcloud auth configure-docker --quiet

# Create .env from config.env if it doesn't exist
if [ ! -f .env ]; then
    echo ""
    echo "Creating .env configuration file..."
    cp config.env .env
    echo "✅ Created .env from config.env"
    echo ""
    echo "⚠️  IMPORTANT: Edit .env to configure your project settings:"
    echo "   - PROJECT_ID: Your GCP project ID"
    echo "   - SERVICE_ACCOUNT: Your service account email"
    echo "   - BUCKET_NAME: Your GCS bucket name"
    echo "   - INPUT_PATH: Path to input data in bucket"
    echo "   - OUTPUT_PATH: Path for output data in bucket"
    echo ""
    echo "   Edit with: nano .env"
else
    echo "✅ .env file already exists"
fi

# Make scripts executable
chmod +x *.sh
echo "✅ Made all scripts executable"

echo ""
echo "========================================="
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env with your project configuration"
echo "2. Ensure your input data is in GCS"
echo "3. Run: ./submit_batch_job.sh"
echo "========================================="