# Cloud Batch GPU Pipeline for Calibration Optimization

This pipeline runs the L0 calibration optimization on GCP using Cloud Batch with GPU support.

## Architecture
- **Cloud Batch**: Automatically provisions GPU VMs, runs the job, and tears down
- **Spot Instances**: Uses spot pricing for cost efficiency
- **GPU**: NVIDIA Tesla P100 for CUDA acceleration
- **Auto-shutdown**: VM terminates after job completion

## Quick Start

### For You (Original User)

```bash
cd batch_pipeline
./submit_batch_job.sh
```

Your settings are already configured in `config.env`.

### For Other Users

1. **Run setup script:**
```bash
cd batch_pipeline
./setup.sh
```

2. **Edit configuration:**
```bash
# Copy and edit configuration
cp config.env .env
nano .env
```

Change these settings:
- `PROJECT_ID`: Your GCP project ID
- `SERVICE_ACCOUNT`: Your service account email
- `BUCKET_NAME`: Your GCS bucket name
- `INPUT_PATH`: Path to input data in bucket
- `OUTPUT_PATH`: Path for output data in bucket

3. **Submit the job:**
```bash
./submit_batch_job.sh
```

4. **Monitor progress:**
```bash
./monitor_batch_job.sh <job_name>
```

## Files
- `config.env` - Configuration template with your current settings
- `.env` - User's custom configuration (created from config.env)
- `Dockerfile` - Container with CUDA, PyTorch, L0 package
- `optimize_weights.py` - The optimization script
- `run_batch_job.sh` - Runs inside container
- `generate_config.py` - Creates batch config from .env
- `submit_batch_job.sh` - Builds, pushes, submits job
- `monitor_batch_job.sh` - Monitors job progress
- `setup.sh` - Initial setup for new users

## How It Works

1. `submit_batch_job.sh` reads configuration from `.env` (or `config.env`)
2. Builds Docker image with your code
3. Pushes to Google Container Registry
4. Generates `batch_job_config.json` from your settings
5. Submits job to Cloud Batch
6. Cloud Batch:
   - Provisions spot GPU VM
   - Pulls Docker image
   - Downloads data from GCS
   - Runs optimization
   - Uploads results to GCS
   - Terminates VM

## Monitoring

View job status:
```bash
gcloud batch jobs describe <job_name> --location=us-central1
```

View logs:
```bash
gcloud logging read "resource.type=batch.googleapis.com/Job AND resource.labels.job_id=<job_name>"
```

## Cost Savings
- Spot instances: ~70% cheaper than on-demand
- Auto-shutdown: No forgotten VMs
- P100 GPU: Older but sufficient, cheaper than V100/A100