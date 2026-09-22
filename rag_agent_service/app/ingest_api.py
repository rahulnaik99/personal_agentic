"""
Ingestion runs over plain HTTP (not gRPC) since it's a bulk/admin
operation — uploading files — rather than a conversational "task" in
the AgentService sense. Runs as its own small FastAPI process alongside
the gRPC server (see docker-compose.yml: rag-agent-service exposes both
a gRPC port and this HTTP port).

Every document is tagged with a `category` at ingest time (e.g.
"profession_doc", "financial_doc") — this is what lets retrieval later
filter to the right document set instead of searching everything. See
rag_agent_service/app/agent.py's classify_category().
"""
import json
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, Form, UploadFile
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from rag_agent_service.app.chunking import UNCATEGORIZED
from rag_agent_service.app.ingestion import ingest_file_with_progress, ingest_files, ingest_texts
from rag_agent_service.app.loaders import SUPPORTED_EXTENSIONS
from shared.core.logging import configure_logging
from shared.vectorstore.weaviate_client import get_ingestion_stats, get_weaviate_client

configure_logging()
app = FastAPI(title="rag-agent-ingest")


class IngestTextDocument(BaseModel):
    text: str
    source: str
    category: str = UNCATEGORIZED


class IngestTextRequest(BaseModel):
    documents: list[IngestTextDocument]


@app.post("/ingest/text")
async def ingest_text_endpoint(request: IngestTextRequest):
    triples = [(d.text, d.source, d.category) for d in request.documents]
    return ingest_texts(triples)
    
@app.get("/stats")
async def stats():
    try:
        return get_ingestion_stats()
    except Exception as exc:
        return {"error": str(exc)}

@app.post("/ingest/file")
async def ingest_file_endpoint(file: UploadFile, category: str = Form(default=UNCATEGORIZED)):
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return {"error": f"Unsupported file type {ext}. Supported: {sorted(SUPPORTED_EXTENSIONS)}"}

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        return ingest_files([(tmp_path, file.filename, category)])
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/ingest/file/stream")
async def ingest_file_stream_endpoint(file: UploadFile, category: str = Form(default=UNCATEGORIZED)):
    """
    SSE variant of /ingest/file — emits one event per real ingestion
    stage (loading_file, checking_duplicate, chunking, embedding_and_storing,
    completed) so the UI can render a live checklist/progress bar instead
    of just waiting on a single blocking response.
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        async def _error_stream():
            yield {"event": "error", "data": json.dumps({"error": f"Unsupported file type {ext}. Supported: {sorted(SUPPORTED_EXTENSIONS)}"})}
        return EventSourceResponse(_error_stream())

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    async def _stream():
        try:
            for event in ingest_file_with_progress(tmp_path, file.filename, category):
                yield {"event": event["stage"], "data": json.dumps(event)}
        except Exception as exc:
            yield {"event": "error", "data": json.dumps({"error": str(exc)})}
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    return EventSourceResponse(_stream())


@app.get("/health")
async def health():
    try:
        client = get_weaviate_client()
        weaviate_ready = client.is_ready()
    except Exception:
        weaviate_ready = False
    return {"status": "ok", "service": "rag-agent-ingest", "weaviate_ready": weaviate_ready}


@app.get("/stats")
async def stats():
    """Powers the Streamlit ingestion dashboard (live Weaviate size, category
    breakdown, active/inactive chunk counts)."""
    return get_ingestion_stats()
