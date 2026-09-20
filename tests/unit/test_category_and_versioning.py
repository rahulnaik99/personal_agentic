from rag_agent_service.app.chunking import UNCATEGORIZED, chunk_document, chunk_elements
from rag_agent_service.app.ingestion import _hash_content, _resolve_action
from rag_agent_service.app.loaders.base import DocElement


def test_chunk_document_defaults_to_uncategorized():
    result = chunk_document("some text", source="test.txt")
    assert result.parents[0].category == UNCATEGORIZED
    assert result.children[0].category == UNCATEGORIZED


def test_chunk_document_applies_given_category():
    result = chunk_document("resume content here", source="resume.txt", category="profession_doc")
    assert all(p.category == "profession_doc" for p in result.parents)
    assert all(c.category == "profession_doc" for c in result.children)


def test_chunk_elements_applies_category_to_table_and_image():
    elements = [
        DocElement(type="table", text="<table></table>"),
        DocElement(type="image", text="a chart", image_path="/tmp/x.png"),
    ]
    result = chunk_elements(elements, source="doc.pdf", category="financial_doc")
    assert all(c.category == "financial_doc" for c in result.children)


def test_hash_content_deterministic_for_same_text():
    assert _hash_content("hello world") == _hash_content("hello world")


def test_hash_content_differs_for_different_text():
    assert _hash_content("hello world") != _hash_content("hello there")


def test_hash_content_handles_bytes_and_str_equivalently():
    assert _hash_content("hello") == _hash_content(b"hello")


def test_resolve_action_first_ingest_when_no_existing_hash(monkeypatch):
    monkeypatch.setattr("rag_agent_service.app.ingestion.get_existing_chunk_hash", lambda source: None)
    assert _resolve_action("new-doc", "somehash") == "first_ingest"


def test_resolve_action_unchanged_when_hash_matches(monkeypatch):
    monkeypatch.setattr("rag_agent_service.app.ingestion.get_existing_chunk_hash", lambda source: "abc123")
    assert _resolve_action("existing-doc", "abc123") == "unchanged"


def test_resolve_action_updated_when_hash_differs(monkeypatch):
    monkeypatch.setattr("rag_agent_service.app.ingestion.get_existing_chunk_hash", lambda source: "old_hash")
    assert _resolve_action("existing-doc", "new_hash") == "updated"
