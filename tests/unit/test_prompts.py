import textwrap

from shared.core.prompts import load_prompt


def test_load_prompt_strips_frontmatter(tmp_path):
    content = textwrap.dedent("""\
        ---
        name: example
        output_schema: plain_text
        ---
        # Role

        You are a helpful assistant.
        """)
    path = tmp_path / "example.md"
    path.write_text(content, encoding="utf-8")

    load_prompt.cache_clear()
    result = load_prompt(str(path))

    assert "---" not in result
    assert "name: example" not in result
    assert result.startswith("# Role")


def test_load_prompt_no_frontmatter_returns_full_text(tmp_path):
    content = "Just a plain prompt with no frontmatter."
    path = tmp_path / "plain.md"
    path.write_text(content, encoding="utf-8")

    load_prompt.cache_clear()
    result = load_prompt(str(path))
    assert result == content


def test_load_prompt_is_cached(tmp_path):
    path = tmp_path / "cached.md"
    path.write_text("original content", encoding="utf-8")
    load_prompt.cache_clear()

    first = load_prompt(str(path))
    path.write_text("changed content", encoding="utf-8")
    second = load_prompt(str(path))  # same path -> cached, should NOT pick up the change

    assert first == second == "original content"
