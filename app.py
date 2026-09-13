import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate

from gcs_utils import faiss_index_exists_locally, download_faiss_index
from ingest import load_local_pdfs, load_pdfs_from_gcs, run_ingestion, DATA_FOLDER

# -----------------------------------------
# ENV SETUP
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
    st.session_state.kb_ready = faiss_index_exists_locally(VECTOR_DB_PATH)
if "kb_checked_gcs" not in st.session_state:
    st.session_state.kb_checked_gcs = False

if not st.session_state.kb_ready and not st.session_state.kb_checked_gcs:
    st.session_state.kb_checked_gcs = True
    try:
        download_faiss_index(VECTOR_DB_PATH, DEFAULT_BUCKET, DEFAULT_PREFIX)
        st.session_state.kb_ready = True
    except Exception:
        pass  # no index in GCS yet — ingestion UI below will build one


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
You are an HR Support Assistant for company employees.

Answer questions using ONLY the information provided in the HR documents.
If the answer is not present, say:
"I’m not sure based on current HR policies."

Be clear, professional, and concise.

HR Context:
{context}

Employee Question:
{question}
""")


# -----------------------------------------
# INGESTION UI
# -----------------------------------------
def render_ingestion_panel(expanded: bool):
    container = st.expander("📚 Knowledge Base Setup", expanded=expanded) if not expanded else st.container()

    with container:
        if not expanded:
            st.markdown("Rebuild the index if you've added or changed HR documents.")

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

                run_ingestion(
                    documents,
                    local_path=VECTOR_DB_PATH,
                    upload_to_gcs=upload_to_gcs,
                    bucket=bucket,
                    prefix=prefix,
                    progress_callback=progress_callback,
                )

                status_box.update(label="Knowledge base ready ✅", state="complete")

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
st.title("HR Support Chatbot")

if not st.session_state.kb_ready:
    st.markdown(
        "No knowledge base found yet. Build one below — this only needs to be "
        "done once (or whenever HR documents change)."
    )
    render_ingestion_panel(expanded=True)
    st.stop()

render_ingestion_panel(expanded=False)

st.markdown(
    "Ask questions about onboarding, leave policy, compensation, IT usage, "
    "security guidelines, and workplace safety."
)

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
llm = load_llm()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_question = st.chat_input("Ask an HR-related question...")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.spinner("Searching HR policies..."):
        docs = retriever.invoke(user_question)
        context = "\n\n".join(doc.page_content for doc in docs)

        response = llm.invoke(
            PROMPT.format_messages(
                context=context,
                question=user_question
            )
        )

    with st.chat_message("assistant"):
        st.markdown(response.content)

    st.session_state.messages.append({"role": "assistant", "content": response.content})
