import os
import shutil
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.prompts import ChatPromptTemplate

from gcs_utils import faiss_index_exists_locally, download_faiss_index
from ingest import load_local_pdfs, load_pdfs_from_gcs, run_ingestion, DATA_FOLDER

# -----------------------------------------
# ENV
# -----------------------------------------
load_dotenv()

st.set_page_config(
    page_title="HR Support Chatbot",
    page_icon="💼",
    layout="centered"
)

# -----------------------------------------
# CONSTANTS
# -----------------------------------------
VECTOR_DB_PATH = "hr_faiss_index"
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")
DEFAULT_BUCKET = os.getenv("GCS_BUCKET_NAME", "har-rag")
DEFAULT_PREFIX = os.getenv("GCS_INDEX_PREFIX", "hr-faiss-index")

# -----------------------------------------
# SESSION STATE
# -----------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "kb_ready" not in st.session_state:
    # A local index (baked into the image, or left over from a previous
    # ingestion run in this container) counts as ready without asking.
    st.session_state.kb_ready = faiss_index_exists_locally(VECTOR_DB_PATH)
if "kb_checked_gcs" not in st.session_state:
    st.session_state.kb_checked_gcs = False

# If no local index yet, try a silent one-time pull from GCS in case a
# previous ingestion run already published one there.
if not st.session_state.kb_ready and not st.session_state.kb_checked_gcs:
    st.session_state.kb_checked_gcs = True
    try:
        download_faiss_index(VECTOR_DB_PATH, DEFAULT_BUCKET, DEFAULT_PREFIX)
        st.session_state.kb_ready = True
    except Exception:
        # No index in GCS yet (or no access) — that's fine, the ingestion
        # UI below will let the user build one.
        pass

# -----------------------------------------
# CACHED RESOURCES
# -----------------------------------------
@st.cache_resource
def load_vectorstore():
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=os.environ["GOOGLE_API_KEY"],
    )
    return FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )


@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        temperature=0,
        google_api_key=os.environ["GOOGLE_API_KEY"],
    )


PROMPT = ChatPromptTemplate.from_template("""
You are an HR Support Assistant.

Use:
1. HR policy context to answer factually
2. Conversation history to understand follow-up questions

Rules:
- Answer ONLY using HR policy context
- If unsure, say: "I’m not sure based on current HR policies."
- Do NOT invent information

Conversation History:
{chat_history}

HR Policy Context:
{context}

Employee Question:
{question}
""")


# -----------------------------------------
# INGESTION UI (rendered inline, either as the whole page — before a
# knowledge base exists — or as a collapsed "rebuild" panel afterward)
# -----------------------------------------
def render_ingestion_panel(expanded: bool):
    container = st.expander("📚 Knowledge Base Setup", expanded=expanded) if not expanded else st.container()

    with container:
        if not expanded:
            st.markdown(
                "Rebuild the index if you've added or changed HR documents."
            )

        source = st.radio(
            "Source PDFs",
            options=["Use bundled documents/ folder", "Upload PDFs now", "Load from a GCS folder"],
            key="ingest_source",
        )

        uploaded_files = None
        gcs_docs_prefix = None

        if source == "Upload PDFs now":
            uploaded_files = st.file_uploader(
                "Upload HR policy PDFs",
                type=["pdf"],
                accept_multiple_files=True,
                key="ingest_uploader",
            )
        elif source == "Load from a GCS folder":
            gcs_docs_prefix = st.text_input(
                "GCS prefix containing source PDFs",
                value="hr-source-docs",
                key="ingest_gcs_docs_prefix",
            )

        with st.expander("Advanced: GCS index storage settings", expanded=False):
            upload_to_gcs = st.checkbox(
                "Publish the built index to GCS (recommended so Cloud Run instances can reuse it)",
                value=True,
                key="ingest_upload_to_gcs",
            )
            bucket = st.text_input("Bucket", value=DEFAULT_BUCKET, key="ingest_bucket")
            prefix = st.text_input("Prefix", value=DEFAULT_PREFIX, key="ingest_prefix")

        run_clicked = st.button("🔄 Build Knowledge Base", type="primary", key="ingest_run_button")

        if run_clicked:
            status_box = st.status("Starting ingestion...", expanded=True)

            def progress_callback(message: str):
                status_box.write(message)

            try:
                # 1. Gather source documents
                if source == "Use bundled documents/ folder":
                    if not os.path.isdir(DATA_FOLDER) or not any(
                        f.endswith(".pdf") for f in os.listdir(DATA_FOLDER)
                    ):
                        status_box.update(label="No PDFs found", state="error")
                        st.error(f"No PDFs found in ./{DATA_FOLDER}/. Add files there, or choose a different source.")
                        st.stop()
                    status_box.write(f"Loading PDFs from ./{DATA_FOLDER}/ ...")
                    documents = load_local_pdfs(DATA_FOLDER)

                elif source == "Upload PDFs now":
                    if not uploaded_files:
                        status_box.update(label="No files uploaded", state="error")
                        st.error("Please upload at least one PDF file.")
                        st.stop()
                    Path(DATA_FOLDER).mkdir(parents=True, exist_ok=True)
                    for f in uploaded_files:
                        dest = Path(DATA_FOLDER) / f.name
                        with open(dest, "wb") as out:
                            out.write(f.getbuffer())
                    status_box.write(f"Saved {len(uploaded_files)} uploaded file(s) to ./{DATA_FOLDER}/")
                    documents = load_local_pdfs(DATA_FOLDER)

                else:  # Load from a GCS folder
                    if not gcs_docs_prefix:
                        status_box.update(label="Missing GCS prefix", state="error")
                        st.error("Please provide a GCS prefix to load PDFs from.")
                        st.stop()
                    status_box.write(f"Downloading PDFs from gs://{bucket}/{gcs_docs_prefix}/ ...")
                    documents = load_pdfs_from_gcs(bucket, gcs_docs_prefix)

                # 2. Run the shared ingestion pipeline (split -> embed -> build -> save -> upload)
                run_ingestion(
                    documents,
                    local_path=VECTOR_DB_PATH,
                    upload_to_gcs=upload_to_gcs,
                    bucket=bucket,
                    prefix=prefix,
                    progress_callback=progress_callback,
                )

                status_box.update(label="Knowledge base ready ✅", state="complete")

                # Invalidate any previously-cached (now stale) vectorstore and move to chat
                load_vectorstore.clear()
                st.session_state.kb_ready = True
                st.success("Knowledge base built successfully. Loading chat...")
                st.rerun()

            except Exception as e:
                status_box.update(label="Ingestion failed", state="error")
                st.error(f"Ingestion failed: {e}")


# -----------------------------------------
# PAGE
# -----------------------------------------
st.title("💼 HR Support Chatbot (Memory Enabled)")

if not st.session_state.kb_ready:
    st.markdown(
        "No knowledge base found yet. Build one below — this only needs to be "
        "done once (or whenever HR documents change)."
    )
    render_ingestion_panel(expanded=True)
    st.stop()  # don't render chat until a knowledge base exists

# Knowledge base is ready — show a collapsed rebuild panel, then chat.
render_ingestion_panel(expanded=False)

st.markdown("Ask HR-related questions. Follow-up questions are supported.")

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
llm = load_llm()

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_question = st.chat_input("Ask an HR-related question...")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.spinner("Thinking..."):
        history = st.session_state.messages[-6:]
        chat_history = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history)

        docs = retriever.invoke(user_question)
        context = "\n\n".join(doc.page_content for doc in docs)

        response = llm.invoke(
            PROMPT.format_messages(
                chat_history=chat_history,
                context=context,
                question=user_question,
            )
        )

    with st.chat_message("assistant"):
        st.markdown(response.content)

    st.session_state.messages.append({"role": "assistant", "content": response.content})
