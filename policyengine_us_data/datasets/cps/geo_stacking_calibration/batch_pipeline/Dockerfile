# Use Google's Deep Learning container (optimized for GCP)
# Has PyTorch 2.x, CUDA, cuDNN, numpy, scipy, pandas, and gsutil pre-installed
FROM gcr.io/deeplearning-platform-release/pytorch-gpu.2-0:latest

# Fix NumPy compatibility issue - force reinstall numpy compatible version
RUN pip install --no-cache-dir --force-reinstall "numpy>=1.24,<2.0"

# Install additional dependencies
RUN pip install --no-cache-dir \
    google-cloud-storage

# Install L0 package from GitHub (this might have compiled components)
RUN pip install --no-cache-dir --no-build-isolation git+https://github.com/PolicyEngine/L0.git@L0-sept

# Create working directory
WORKDIR /app

# Copy the optimization script
COPY optimize_weights.py /app/
COPY run_batch_job.sh /app/

# Make the run script executable
RUN chmod +x /app/run_batch_job.sh

# Set the entrypoint
ENTRYPOINT ["/app/run_batch_job.sh"]
