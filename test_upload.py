
from google.cloud import storage

storage_client = storage.Client()
bucket = storage_client.bucket("policyengine-us-data")
blob = "README.md"
blob = bucket.blob(blob)
blob.upload_from_filename("README.md")