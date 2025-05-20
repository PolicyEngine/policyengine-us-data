
from google.cloud import storage
import google.auth

credentials, project_id = google.auth.default()

storage_client = storage.Client(credentials=credentials, project=project_id)
bucket = storage_client.bucket("policyengine-us-data")
blob = "README.md"
blob = bucket.blob(blob)
blob.upload_from_filename("README.md")
