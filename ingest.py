"""
ingest.py
---------
Builds the FAISS knowledge base from HR PDFs, using Gemini embeddings.

This module is written to be used two ways:

1. As a CLI script (for Cloud Shell / local / CI use):
       python ingest.py
       python ingest.py --upload-to-gcs
       python ingest.py --docs-gcs-prefix hr-source-docs --upload-to-gcs

2. As a library imported by the Streamlit apps, so ingestion can be
   triggered from a button in the UI instead of a terminal. See
   `run_ingestion()` below — the apps call this directly with a
   `progress_callback` to update an on-screen status widget.

Auth:
- Gemini: reads GOOGLE_API_KEY from the environment.
- GCS: uses Application Default Credentials (Cloud Shell's logged-in
  gcloud identity locally, or the Cloud Run service's runtime service
  account in production). No key file needed either way.
"""

import argparse
import os
import tempfile
from pathlib import Path
from typing import Callable, List, Optional

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

DATA_FOLDER = "documents"
VECTOR_DB_PATH = "hr_faiss_index"
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")

ProgressFn = Optional[Callable[[str], None]]


def _report(progress_callback: ProgressFn, message: str):
    print(message)
    if progress_callback:
        progress_callback(message)


# -----------------------------------------
# LOADING
# -----------------------------------------
def load_local_pdfs(folder: str) -> List:
    documents = []
    for file in sorted(os.listdir(folder)):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder, file))
            documents.extend(loader.load())
    return documents


def load_pdfs_from_gcs(bucket_name: str, prefix: str) -> List:
    """Download source PDFs from a GCS prefix into a temp dir, then load them.
    Useful when running in Cloud Shell (or Cloud Run) with no local
    `documents/` folder checked out."""
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    tmp_dir = tempfile.mkdtemp(prefix="hr_docs_")

    blobs = list(bucket.list_blobs(prefix=prefix.strip("/")))
    pdf_blobs = [b for b in blobs if b.name.lower().endswith(".pdf")]
    if not pdf_blobs:
        raise FileNotFoundError(f"No PDFs found under gs://{bucket_name}/{prefix}/")

    for blob in pdf_blobs:
        local_path = Path(tmp_dir) / Path(blob.name).name
        blob.download_to_filename(str(local_path))
        print(f"Downloaded gs://{bucket_name}/{blob.name} -> {local_path}")

    return load_local_pdfs(tmp_dir)


# -----------------------------------------
# SPLIT / EMBED / BUILD
# -----------------------------------------
def split_documents(documents: List, chunk_size: int = 800, chunk_overlap: int = 150) -> List:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return text_splitter.split_documents(documents)


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=os.environ["GOOGLE_API_KEY"],
    )


def build_index(chunks: List, embeddings: GoogleGenerativeAIEmbeddings) -> FAISS:
    return FAISS.from_documents(chunks, embeddings)


# -----------------------------------------
# END-TO-END PIPELINE (used by both the CLI and the Streamlit UI)
# -----------------------------------------
def run_ingestion(
    documents: List,
    local_path: str = VECTOR_DB_PATH,
    upload_to_gcs: bool = False,
    bucket: Optional[str] = None,
    prefix: Optional[str] = None,
    progress_callback: ProgressFn = None,
) -> FAISS:
    """
    Full pipeline: split -> embed -> build FAISS -> save locally -> (optional) upload to GCS.
    Returns the built FAISS vectorstore.
    """
    _report(progress_callback, f"Loaded {len(documents)} PDF page(s).")

    chunks = split_documents(documents)
    _report(progress_callback, f"Created {len(chunks)} text chunk(s).")

    _report(progress_callback, "Generating Gemini embeddings and building the FAISS index...")
    embeddings = get_embeddings()
    vectorstore = build_index(chunks, embeddings)

    _report(progress_callback, f"Saving index to ./{local_path} ...")
    vectorstore.save_local(local_path)

    if upload_to_gcs:
        from gcs_utils import upload_faiss_index
        _report(progress_callback, "Uploading index to Google Cloud Storage...")
        upload_faiss_index(local_path, bucket_name=bucket, prefix=prefix)
        _report(progress_callback, "Upload complete.")

    _report(progress_callback, "Knowledge base ready.")
    return vectorstore


# -----------------------------------------
# CLI ENTRYPOINT
# -----------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Ingest HR PDFs into a FAISS index.")
    parser.add_argument(
        "--docs-gcs-prefix",
        default=None,
        help="If set, load source PDFs from this prefix in the GCS bucket "
             "instead of the local documents/ folder.",
    )
    parser.add_argument(
        "--upload-to-gcs",
        action="store_true",
        help="After building the index locally, upload it to GCS so Cloud Run can use it.",
    )
    parser.add_argument(
        "--bucket",
        default=os.getenv("GCS_BUCKET_NAME", "har-rag"),
        help="GCS bucket name (default: env GCS_BUCKET_NAME or 'har-rag').",
    )
    parser.add_argument(
        "--prefix",
        default=os.getenv("GCS_INDEX_PREFIX", "hr-faiss-index"),
        help="Folder/prefix inside the bucket to store the FAISS index "
             "(default: env GCS_INDEX_PREFIX or 'hr-faiss-index').",
    )
    args = parser.parse_args()

    if args.docs_gcs_prefix:
        documents = load_pdfs_from_gcs(args.bucket, args.docs_gcs_prefix)
    else:
        documents = load_local_pdfs(DATA_FOLDER)

    run_ingestion(
        documents,
        upload_to_gcs=args.upload_to_gcs,
        bucket=args.bucket,
        prefix=args.prefix,
    )


if __name__ == "__main__":
    main()
