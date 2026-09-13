"""
gcs_utils.py
------------
Small helper module used by ingest.py and the Streamlit apps to move the
FAISS index folder (index.faiss + index.pkl) to/from a Google Cloud
Storage bucket.

Why this exists:
Cloud Run containers have an ephemeral, read-only-by-default filesystem
(the writable layer disappears every time a new instance starts / scales
to zero). So the FAISS index cannot simply live on local disk in
production the way it does when you run everything on a laptop:

    1. `ingest.py` builds the index locally, then UPLOADS it to GCS.
    2. `app.py` / `app_with_memory.py` DOWNLOAD the index from GCS into
       the container's local /tmp (or the working dir) at startup,
       then load it with FAISS.load_local() as before.

All bucket/prefix values are read from environment variables so nothing
is hard-coded:

    GCS_BUCKET_NAME   e.g. "har-rag"
    GCS_INDEX_PREFIX  e.g. "hr-faiss-index"   (folder/prefix inside the bucket)

Auth:
- Locally / in Cloud Shell: uses Application Default Credentials, i.e.
  whatever you get from `gcloud auth application-default login` or the
  Cloud Shell's built-in credentials. No key file needed.
- On Cloud Run: uses the service's attached runtime service account
  automatically. No key file needed there either.
"""

import os
from pathlib import Path

from google.cloud import storage

INDEX_FILES = ["index.faiss", "index.pkl"]


def _bucket_and_prefix(bucket_name: str | None = None, prefix: str | None = None):
    bucket_name = bucket_name or os.getenv("GCS_BUCKET_NAME", "har-rag")
    prefix = prefix or os.getenv("GCS_INDEX_PREFIX", "hr-faiss-index")
    # normalize: no leading/trailing slashes
    prefix = prefix.strip("/")
    return bucket_name, prefix


def upload_faiss_index(local_dir: str, bucket_name: str | None = None, prefix: str | None = None):
    """Upload index.faiss + index.pkl from local_dir to gs://<bucket>/<prefix>/"""
    bucket_name, prefix = _bucket_and_prefix(bucket_name, prefix)
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    for fname in INDEX_FILES:
        local_path = Path(local_dir) / fname
        if not local_path.exists():
            raise FileNotFoundError(f"Expected index file not found: {local_path}")
        blob_path = f"{prefix}/{fname}"
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(str(local_path))
        print(f"Uploaded {local_path} -> gs://{bucket_name}/{blob_path}")


def download_faiss_index(local_dir: str, bucket_name: str | None = None, prefix: str | None = None):
    """Download index.faiss + index.pkl from gs://<bucket>/<prefix>/ into local_dir."""
    bucket_name, prefix = _bucket_and_prefix(bucket_name, prefix)
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    Path(local_dir).mkdir(parents=True, exist_ok=True)

    for fname in INDEX_FILES:
        blob_path = f"{prefix}/{fname}"
        blob = bucket.blob(blob_path)
        local_path = Path(local_dir) / fname
        blob.download_to_filename(str(local_path))
        print(f"Downloaded gs://{bucket_name}/{blob_path} -> {local_path}")


def faiss_index_exists_locally(local_dir: str) -> bool:
    return all((Path(local_dir) / fname).exists() for fname in INDEX_FILES)


def ensure_faiss_index(local_dir: str, bucket_name: str | None = None, prefix: str | None = None):
    """
    Make sure a usable FAISS index exists at local_dir.
    If it's already there (e.g. bundled in the image or mounted), do nothing.
    Otherwise, try to pull it down from GCS. Raises if neither is available.
    """
    if faiss_index_exists_locally(local_dir):
        return

    bucket_name, prefix = _bucket_and_prefix(bucket_name, prefix)
    print(f"Local FAISS index not found at '{local_dir}'. Downloading from "
          f"gs://{bucket_name}/{prefix}/ ...")
    download_faiss_index(local_dir, bucket_name, prefix)
