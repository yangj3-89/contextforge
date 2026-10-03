import json

import pytest

from app.ingestion.parsers import (
    ParseError,
    UnsupportedFormatError,
    detect_source_type,
    parse_document,
)


def test_detect_source_type():
    assert detect_source_type("a.MD") == "markdown"
    assert detect_source_type("a.json") == "json"
    with pytest.raises(UnsupportedFormatError):
        detect_source_type("a.docx")


def test_txt_paragraph_blocks_have_exact_offsets():
    doc = parse_document("notes.txt", b"First para line one.\nline two.\n\nSecond para.")
    assert doc.source_type == "txt"
    assert len(doc.blocks) == 2
    for block in doc.blocks:
        assert doc.text[block.char_start : block.char_end] == block.text


def test_markdown_keeps_headings_in_text_and_tracks_heading_path():
    md = b"# Title\n\nIntro.\n\n## Section A\n\nBody A.\n\n### Sub\n\nBody sub.\n\n```\ncode block\n\nstill code\n```\n"
    doc = parse_document("doc.md", md)
    assert doc.title == "Title"
    assert "## Section A" in doc.text
    sub = [b for b in doc.blocks if "Body sub." in b.text][0]
    assert sub.metadata["heading_path"] == ["Title", "Section A", "Sub"]
    heading_a = [b for b in doc.blocks if b.text.startswith("## Section A")][0]
    assert heading_a.breaks_before
    code = [b for b in doc.blocks if "code block" in b.text][0]
    assert "still code" in code.text  # fenced code is not split on blank lines


def test_json_records_become_blocks_with_paths():
    payload = {"title": "Tickets", "project": "HAL", "issues": [{"key": "HAL-1", "meta": {"owner": "Ana"}}, {"key": "HAL-2"}]}
    doc = parse_document("t.json", json.dumps(payload).encode())
    assert doc.title == "Tickets"
    kinds = [b.metadata.get("kind") for b in doc.blocks]
    assert kinds == ["header", "record", "record"]
    assert doc.blocks[1].metadata["json_path"] == "/issues/0"
    assert "meta.owner: Ana" in doc.blocks[1].text
    assert all(b.breaks_before for b in doc.blocks[1:])


def test_json_top_level_list():
    doc = parse_document("l.json", b'[{"a": 1}, {"a": 2}]')
    assert [b.metadata["json_path"] for b in doc.blocks] == ["/0", "/1"]


def test_invalid_json_raises():
    with pytest.raises(ParseError):
        parse_document("bad.json", b"{not json")


def test_empty_document_raises():
    with pytest.raises(ParseError):
        parse_document("empty.txt", b"   \n  ")


def test_pdf_pages_are_tracked():
    fpdf = pytest.importorskip("fpdf")
    pdf = fpdf.FPDF()
    for text in ("Page one talks about retrieval.", "Page two talks about abstention."):
        pdf.add_page()
        pdf.set_font("Helvetica", size=11)
        pdf.multi_cell(0, 6, text)
    doc = parse_document("r.pdf", bytes(pdf.output()))
    assert doc.source_type == "pdf"
    pages = {b.metadata["page"] for b in doc.blocks}
    assert pages == {1, 2}
    assert "abstention" in doc.text


def test_corrupt_pdf_raises_parse_error():
    with pytest.raises(ParseError):
        parse_document("x.pdf", b"%PDF-1.4 garbage")
