from rag_agent_service.app.chunking import chunk_document, chunk_elements
from rag_agent_service.app.loaders.base import DocElement


def test_chunk_document_produces_parent_and_child():
    text = "This is a test document. " * 200  # long enough to force multiple chunks
    result = chunk_document(text, source="test.txt")
    assert len(result.parents) >= 1
    assert len(result.children) >= 1
    # every child must reference a real parent id
    parent_ids = {p.id for p in result.parents}
    assert all(c.parent_id in parent_ids for c in result.children)


def test_chunk_document_short_text_single_chunk():
    result = chunk_document("short text.", source="test.txt")
    assert len(result.parents) == 1
    assert len(result.children) == 1
    assert result.parents[0].element_type == "text"


def test_table_element_kept_whole_not_split():
    table_html = "<table>" + "<tr><td>row</td></tr>" * 50 + "</table>"
    elements = [DocElement(type="table", text=table_html)]
    result = chunk_elements(elements, source="doc.pdf")

    assert len(result.parents) == 1
    assert len(result.children) == 1
    assert result.children[0].element_type == "table"
    assert result.children[0].text == table_html  # unsplit, exact match


def test_image_element_kept_whole_with_caption():
    elements = [DocElement(type="image", text="A bar chart showing quarterly revenue.", image_path="/tmp/img.png")]
    result = chunk_elements(elements, source="doc.pdf")

    assert len(result.children) == 1
    assert result.children[0].element_type == "image"
    assert result.children[0].image_path == "/tmp/img.png"
    assert result.children[0].text == "A bar chart showing quarterly revenue."


def test_mixed_elements_text_and_table_both_present():
    elements = [
        DocElement(type="text", text="Intro paragraph about the report."),
        DocElement(type="table", text="<table><tr><td>1</td></tr></table>"),
        DocElement(type="text", text="Concluding paragraph."),
    ]
    result = chunk_elements(elements, source="doc.pdf")

    element_types = {c.element_type for c in result.children}
    assert "table" in element_types
    assert "text" in element_types
    # exactly one table child, unsplit
    table_children = [c for c in result.children if c.element_type == "table"]
    assert len(table_children) == 1


def test_empty_elements_produces_empty_result():
    result = chunk_elements([], source="doc.pdf")
    assert result.parents == []
    assert result.children == []
