# HR Support Chatbot (RAG-Based, Google Gemini + Cloud Run)

## Overview

This project implements a **Retrieval-Augmented Generation (RAG)** based **HR Support Chatbot** that answers employee questions using official company HR documents.

It runs on **Google Gemini** for both the chat model and the embedding model, stores its FAISS index in **Google Cloud Storage**, and is set up to deploy to **Google Cloud Run**.

Ingestion — building the FAISS knowledge base from the HR PDFs — is **built into the Streamlit UI itself**. There's no separate terminal step required: the app shows a "Knowledge Base Setup" screen first, and only unlocks the chat once an index exists (either bundled with the app, pulled from GCS, or just built by you in the browser).

---

## What changed from the OpenAI version

* `ChatOpenAI` → `ChatGoogleGenerativeAI` (`gemini-2.5-flash`)
* `OpenAIEmbeddings` → `GoogleGenerativeAIEmbeddings` (`models/gemini-embedding-001`)
* **Ingestion moved into the app.** `ingest.py` is now a library of functions (`load_local_pdfs`, `load_pdfs_from_gcs`, `run_ingestion`, ...) that both the CLI and the Streamlit apps call directly — you can still run it from a terminal/Cloud Shell if you prefer, but it's no longer required
* Added `gcs_utils.py` — uploads/downloads the FAISS index to/from a GCS bucket, since Cloud Run's filesystem is ephemeral and can't be relied on to keep a locally-built index between deploys/instances
* Added `Dockerfile`, `app.yaml` (Cloud Run service manifest), and `.github/workflows/docker.yml` (CI/CD) for deploying to Cloud Run
* Removed the old `.env` (it had a real API key committed in it — see **Security note** below)

---

## How the "Knowledge Base first, then Chat" flow works

Every time the app starts, it checks for a usable index in this order:

1. **Local `hr_faiss_index/` folder** — already there? (e.g. baked into the Docker image, or built earlier in this same container) → go straight to chat.
2. **GCS** — not found locally, but one exists at `gs://<bucket>/<prefix>/`? → silently download it, then go to chat.
3. **Neither** → show the **Knowledge Base Setup** panel and stop. You pick a source PDF, click "Build Knowledge Base", watch the live progress log (load → split → embed → save → publish to GCS), and the app automatically flips over to the chat screen when it's done.

Once a knowledge base exists, the same setup panel is still available — collapsed, at the top of the chat screen — so you (or an HR admin) can rebuild it later after documents change, without touching a terminal.

Ingestion source options in the UI:

| Option | What it does |
|---|---|
| **Use bundled `documents/` folder** | Uses whatever PDFs shipped in the image / repo |
| **Upload PDFs now** | Drag-and-drop PDFs straight into the browser; they're saved into `documents/` and ingested |
| **Load from a GCS folder** | Pulls PDFs from a `gs://<bucket>/<prefix>/` you specify — handy for keeping the source-of-truth docs outside the container image |

There's also an "Advanced" toggle to publish the resulting index to GCS (on by default) and to override the bucket/prefix — useful if you're testing against a scratch bucket.

---

## Architecture

```
Employee Question
      ↓
Embedding (Gemini: models/gemini-embedding-001)
      ↓
Vector Similarity Search (FAISS, built/loaded via the app's Knowledge Base panel)
      ↓
Relevant HR Policy Chunks
      ↓
Prompt Augmentation
      ↓
LLM Answer (Gemini: gemini-2.5-flash)
```

---

## Tech Stack

* **Python**
* **Google Gemini API** (via `langchain-google-genai`)

  * `gemini-2.5-flash` (chat model)
  * `models/gemini-embedding-001` (embedding model)
* **LangChain**
* **FAISS** (vector database)
* **Google Cloud Storage** (index storage for Cloud Run)
* **Streamlit** (UI, including the ingestion step)
* **PyPDF** (PDF parsing)
* **Google Cloud Run** (hosting)

---

## Project Structure

```
15_HR_rag_chatbot_memory_UI/
│
├── documents/                 # Bundled source HR PDFs (optional — can also upload via UI or GCS)
├── hr_faiss_index/            # Local FAISS index (created by the in-UI ingestion step)
│
├── ingest.py                  # Ingestion functions — used by the UI, and runnable as a CLI script too
├── gcs_utils.py               # Upload/download the FAISS index to/from GCS
├── app.py                     # Streamlit RAG chatbot (no memory) — Knowledge Base panel + chat
├── app_with_memory.py         # Streamlit RAG chatbot (context-aware) — Knowledge Base panel + chat
├── chatbot.py                 # Console version (expects an index to already exist)
│
├── Dockerfile                 # Container build for Cloud Run
├── app.yaml                   # Cloud Run service manifest
├── .github/workflows/docker.yml  # Optional build + deploy CI/CD
│
├── requirements.txt
├── .env.example                # Copy to .env and fill in real values
└── README.md
```

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GOOGLE_API_KEY` | Yes | — | Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey) |
| `GEMINI_CHAT_MODEL` | No | `gemini-2.5-flash` | Chat model used for answers |
| `GEMINI_EMBEDDING_MODEL` | No | `models/gemini-embedding-001` | Embedding model used for ingestion & retrieval (must match between old and new indexes) |
| `GCS_BUCKET_NAME` | No | `har-rag` | Bucket that stores the FAISS index (and pre-filled as the default in the UI's Advanced panel) |
| `GCS_INDEX_PREFIX` | No | `hr-faiss-index` | Folder/prefix inside the bucket where `index.faiss` / `index.pkl` live |
| `PORT` | No (set by Cloud Run) | `8080` | Port Streamlit binds to |

Copy `.env.example` to `.env` locally and fill in `GOOGLE_API_KEY`:

```bash
cp .env.example .env
```

---

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 2. Run the App

### Without memory

```bash
streamlit run app.py
```

### With context-aware memory (recommended)

```bash
streamlit run app_with_memory.py
```

The first time you run it (with no `hr_faiss_index/` folder and nothing yet in GCS), you'll land on the **Knowledge Base Setup** screen — pick a source and click **Build Knowledge Base**. Once it finishes, the chat UI appears automatically.

---

## 3. (Optional) Build the Index from a Terminal / Cloud Shell Instead

If you'd rather not use the browser step — e.g. for a scripted/CI setup — `ingest.py` still works standalone:

```bash
export GOOGLE_API_KEY=your_gemini_api_key
python ingest.py --upload-to-gcs
```

Cloud Shell already has your `gcloud` identity available as Application Default Credentials, so no key file is needed there either. This uploads `index.faiss` / `index.pkl` to `gs://har-rag/hr-faiss-index/` (override with `--bucket` / `--prefix` or the `GCS_BUCKET_NAME` / `GCS_INDEX_PREFIX` env vars). If your source PDFs live in GCS:

```bash
python ingest.py --docs-gcs-prefix hr-source-docs --upload-to-gcs
```

---

## 4. Bucket & Access Setup (one-time)

Because ingestion can now run **inside** the deployed app, the Cloud Run runtime service account needs both read and write access to the bucket (not just read, as in a download-only setup).

```bash
# create the bucket (skip if it already exists)
gcloud storage buckets create gs://har-rag --location=REGION

# the Cloud Run runtime service account needs to read AND write
# (it downloads an existing index on startup, and can upload a newly-built one from the UI)
gcloud storage buckets add-iam-policy-binding gs://har-rag \
  --member="serviceAccount:PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"

# if you also run ingest.py yourself from Cloud Shell:
gcloud storage buckets add-iam-policy-binding gs://har-rag \
  --member="user:you@example.com" \
  --role="roles/storage.objectAdmin"
```

---

## 5. Deploy to Cloud Run

### Option A — quick deploy straight from source (simplest)

```bash
gcloud run deploy hr-rag-chatbot \
  --source . \
  --region=REGION \
  --allow-unauthenticated \
  --set-env-vars=GCS_BUCKET_NAME=har-rag,GCS_INDEX_PREFIX=hr-faiss-index \
  --set-secrets=GOOGLE_API_KEY=google-api-key:latest
```

(This builds the `Dockerfile` for you via Cloud Build — no need to build/push manually.) After it deploys, open the service URL — if no index exists in GCS yet, you'll see the Knowledge Base Setup screen; build it once there and every future instance will pick it up from GCS automatically.

### Option B — build the image yourself, then deploy with `app.yaml`

```bash
# build & push
gcloud builds submit --tag REGION-docker.pkg.dev/PROJECT_ID/hr-rag-repo/hr-rag-chatbot:latest

# edit app.yaml: fill in PROJECT_ID / REGION in the image path
gcloud run services replace app.yaml --region=REGION
```

### Option C — CI/CD via GitHub Actions

`.github/workflows/docker.yml` builds the image and deploys on every push to `main`. Add these repo secrets first: `GCP_PROJECT_ID`, `GCP_REGION`, `GCP_WORKLOAD_IDP`, `GCP_SERVICE_ACCOUNT`. If you don't use GitHub Actions, you can delete this file — it's optional and Options A/B don't need it.

### Store the Gemini key as a secret (recommended for all options)

```bash
echo -n "YOUR_GEMINI_API_KEY" | gcloud secrets create google-api-key --data-file=-
gcloud secrets add-iam-policy-binding google-api-key \
  --member="serviceAccount:PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

---

## Example Questions

```
What is the onboarding process for new employees?
How many leave days am I entitled to?
Can unused leave be carried forward?
What is the IT security policy on passwords?
What should I do in case of a workplace safety incident?
```

---

## Memory Handling

* Chat history is stored in Streamlit session state
* Previous turns are included in the prompt to support follow-ups
* Retrieval is always performed using the **latest question only** to avoid noise

---

## Design Principles

* **LLM does not act as the source of truth** — HR documents are the authority
* **Retrieval before generation** — answers are grounded in retrieved policy text
* **Fail-safe behavior** — the chatbot refuses to answer when information is missing
* **No hidden setup step** — building the knowledge base is a visible, first-class part of the UI, not a prerequisite script someone has to remember to run
* **Stateless containers** — nothing important lives only on local disk; the index lives in GCS so any Cloud Run instance can start cold and still work

---

## Limitations

* No role-based access control
* No document citations per answer
* Single shared FAISS index (not per-department or per-region)
* Uploading PDFs "now" or ingesting from the bundled folder writes to the container's local disk — on Cloud Run that's ephemeral per-instance, so always leave "Publish the built index to GCS" checked so other instances (and future deploys) can reuse it

---

## Security Note

The original project's `.env` file had a real API key committed to it. That file has been removed from this package — use `.env.example` as a template, and never commit a filled-in `.env` to source control. If that original key is still active anywhere, rotate/revoke it.

---

## Disclaimer

This chatbot is intended for **informational purposes only**. Official HR decisions should always follow company policy and HR approval processes.

## License

Internal / Educational Use Only
