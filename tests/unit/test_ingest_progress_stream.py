import tempfile
from pathlib import Path

from rag_agent_service.app.ingestion import ingest_file_with_progress
from rag_agent_service.app.loaders.base import DocElement


class _FakeCollection:
    pass


class _FakeCollections:
    def get(self, name):
        return _FakeCollection()


class _FakeWeaviateClient:
    collections = _FakeCollections()


def _make_temp_txt(content: str) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
    tmp.write(content.encode("utf-8"))
    tmp.close()
    return tmp.name


def test_ingest_file_with_progress_stops_early_on_unchanged(monkeypatch):
    monkeypatch.setattr("rag_agent_service.app.ingestion.ensure_schema", lambda: None)
    monkeypatch.setattr("rag_agent_service.app.ingestion.get_weaviate_client", lambda: _FakeWeaviateClient())
    monkeypatch.setattr("rag_agent_service.app.ingestion.get_encoder", lambda: object())
    monkeypatch.setattr("rag_agent_service.app.ingestion._resolve_action", lambda source, content_hash: "unchanged")

    path = _make_temp_txt("some content")
    try:
        events = list(ingest_file_with_progress(path, "doc.txt", "profession_doc"))
    finally:
        Path(path).unlink(missing_ok=True)

    stages = [e["stage"] for e in events]
    # Must stop right after checking_duplicate — chunking/embedding never run for a duplicate.
    assert stages == ["loading_file", "loading_file", "checking_duplicate", "checking_duplicate", "completed"]
    assert events[-1]["result"]["status"] == "unchanged"
    assert events[-1]["result"]["parents"] == 0
    assert events[-1]["result"]["children"] == 0


def test_ingest_file_with_progress_runs_all_stages_on_first_ingest(monkeypatch):
    monkeypatch.setattr("rag_agent_service.app.ingestion.ensure_schema", lambda: None)
    monkeypatch.setattr("rag_agent_service.app.ingestion.get_weaviate_client", lambda: _FakeWeaviateClient())
    monkeypatch.setattr("rag_agent_service.app.ingestion.get_encoder", lambda: object())
    monkeypatch.setattr("rag_agent_service.app.ingestion._resolve_action", lambda source, content_hash: "first_ingest")
    monkeypatch.setattr(
        "rag_agent_service.app.ingestion.load_file",
        lambda file_path, source: [DocElement(type="text", text="hello world")],
    )
    monkeypatch.setattr(
        "rag_agent_service.app.ingestion._upsert_chunked_document",
        lambda chunked, content_hash, encoder, pc, cc: (1, 2),
    )

    path = _make_temp_txt("some content")
    try:
        events = list(ingest_file_with_progress(path, "doc.txt", "profession_doc"))
    finally:
        Path(path).unlink(missing_ok=True)

    stages = [e["stage"] for e in events]
    assert stages == [
        "loading_file", "loading_file",
        "checking_duplicate", "checking_duplicate",
        "chunking", "chunking",
        "embedding_and_storing", "embedding_and_storing",
        "completed",
    ]
    assert events[-1]["result"] == {"source": "doc.txt", "status": "first_ingest", "parents": 1, "children": 2}
