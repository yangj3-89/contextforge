from app.core.config import ChunkingSettings
from app.ingestion.chunker import chunk_document
from app.ingestion.parsers import parse_document


def _long_markdown(n_sentences: int) -> bytes:
    body = " ".join(f"Sentence number {i} describes part {i} of the retrieval pipeline." for i in range(n_sentences))
    return f"# Doc\n\n## Long\n\n{body}\n\n## Short\n\nTiny section.\n".encode()


def test_chunks_are_exact_slices_of_document_text():
    doc = parse_document("d.md", _long_markdown(60))
    for spec in chunk_document(doc, ChunkingSettings()):
        assert doc.text[spec.char_start : spec.char_end] == spec.text


def test_long_section_is_split_with_overlap_and_respects_budget():
    settings = ChunkingSettings(max_tokens=60, min_tokens=10, overlap_sentences=1)
    doc = parse_document("d.md", _long_markdown(40))
    specs = [s for s in chunk_document(doc, settings) if s.metadata.get("section") == "Long"]
    assert len(specs) > 3
    for spec in specs:
        assert spec.token_count <= settings.max_tokens + 20  # header block may ride along
    # overlap: consecutive chunks share text
    assert specs[1].char_start < specs[0].char_end


def test_top_level_headings_start_new_chunks():
    doc = parse_document("d.md", _long_markdown(3))
    specs = chunk_document(doc, ChunkingSettings())
    sections = [s.metadata.get("section") for s in specs]
    assert "Short" in sections and "Long" in sections
    short = [s for s in specs if s.metadata.get("section") == "Short"][0]
    assert "Sentence number" not in short.text


def test_search_text_has_contextual_header():
    doc = parse_document("d.md", _long_markdown(3))
    spec = [s for s in chunk_document(doc, ChunkingSettings()) if s.metadata.get("section") == "Short"][0]
    assert spec.search_text.startswith("Doc > Short\n")
    assert spec.search_text.endswith(spec.text)


def test_chunk_indexes_are_sequential():
    doc = parse_document("d.md", _long_markdown(30))
    specs = chunk_document(doc, ChunkingSettings(max_tokens=80))
    assert [s.chunk_index for s in specs] == list(range(len(specs)))
