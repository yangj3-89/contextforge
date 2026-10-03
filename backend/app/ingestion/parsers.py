"""Format-specific parsers.

Each parser turns raw bytes into a :class:`ParsedDocument`: one normalized text
string plus a list of :class:`Block` objects (paragraphs, list groups, JSON
records, PDF paragraphs) with character offsets into that text and structural
metadata (heading path, page number, JSON record path). The chunker only ever
sees blocks, which keeps it format-agnostic.
"""

from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.text import normalize_text

SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".txt": "txt",
    ".text": "txt",
    ".md": "markdown",
    ".markdown": "markdown",
    ".pdf": "pdf",
    ".json": "json",
}


class UnsupportedFormatError(ValueError):
    pass


class ParseError(ValueError):
    pass


@dataclass(slots=True)
class Block:
    text: str
    char_start: int
    char_end: int
    metadata: dict[str, Any] = field(default_factory=dict)
    breaks_before: bool = False
    """Chunker must start a new chunk at this block (e.g. a top-level heading)."""


@dataclass(slots=True)
class ParsedDocument:
    text: str
    blocks: list[Block]
    source_type: str
    title: str
    metadata: dict[str, Any] = field(default_factory=dict)


def detect_source_type(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFormatError(
            f"Unsupported file type '{ext or filename}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    return SUPPORTED_EXTENSIONS[ext]


def parse_document(filename: str, data: bytes) -> ParsedDocument:
    source_type = detect_source_type(filename)
    parser = _PARSERS[source_type]
    parsed = parser(filename, data)
    if not parsed.text.strip():
        raise ParseError(f"No extractable text in '{filename}'")
    return parsed


def _decode(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


def _default_title(filename: str) -> str:
    return Path(filename).stem.replace("_", " ").replace("-", " ").strip()


class _TextBuilder:
    """Accumulates blocks while tracking their offsets in the final document text."""

    def __init__(self) -> None:
        self._parts: list[str] = []
        self._length = 0
        self.blocks: list[Block] = []

    def add(self, text: str, metadata: dict[str, Any], breaks_before: bool = False) -> None:
        text = text.strip()
        if not text:
            return
        if self._parts:
            self._parts.append("\n\n")
            self._length += 2
        start = self._length
        self._parts.append(text)
        self._length += len(text)
        self.blocks.append(Block(text, start, self._length, dict(metadata), breaks_before))

    @property
    def text(self) -> str:
        return "".join(self._parts)


def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


# --------------------------------------------------------------------------- TXT
def parse_txt(filename: str, data: bytes) -> ParsedDocument:
    raw = normalize_text(_decode(data))
    builder = _TextBuilder()
    for para in _paragraphs(raw):
        builder.add(para, {})
    return ParsedDocument(builder.text, builder.blocks, "txt", _default_title(filename))


# ---------------------------------------------------------------------- Markdown
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")


def parse_markdown(filename: str, data: bytes) -> ParsedDocument:
    raw = normalize_text(_decode(data))
    builder = _TextBuilder()
    heading_path: list[tuple[int, str]] = []
    title: str | None = None
    pending_break = False
    buffer: list[str] = []
    in_fence = False

    def flush() -> None:
        nonlocal pending_break
        if buffer:
            meta = {"heading_path": [h for _, h in heading_path]}
            if heading_path:
                meta["section"] = heading_path[-1][1]
            builder.add("\n".join(buffer), meta, breaks_before=pending_break)
            pending_break = False
            buffer.clear()

    for line in raw.split("\n"):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            buffer.append(line)
            continue
        if in_fence:
            buffer.append(line)
            continue
        heading = _HEADING_RE.match(line)
        if heading:
            flush()
            level, heading_text = len(heading.group(1)), heading.group(2).strip()
            if level == 1 and title is None:
                title = heading_text
            heading_path = [(lvl, h) for lvl, h in heading_path if lvl < level]
            heading_path.append((level, heading_text))
            if level <= 2:
                pending_break = True
            # Keep the heading line in the document text (provenance) as the first
            # block of its section; the chunker packs it with the following paragraph.
            buffer.append(line)
            continue
        if not line.strip():
            flush()
            continue
        buffer.append(line)
    flush()
    return ParsedDocument(
        builder.text, builder.blocks, "markdown", title or _default_title(filename)
    )


# --------------------------------------------------------------------------- PDF
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")


def _pdf_page_paragraphs(page_text: str) -> list[str]:
    page_text = _HYPHEN_BREAK_RE.sub(r"\1\2", page_text)
    paragraphs = _paragraphs(page_text)
    if len(paragraphs) <= 1:
        # Many PDF extractors drop blank lines; fall back to line-based grouping:
        # a line ending in terminal punctuation closes a paragraph.
        paragraphs, current = [], []
        for line in page_text.split("\n"):
            line = line.strip()
            if not line:
                continue
            current.append(line)
            if line.endswith((".", "!", "?", ":")):
                paragraphs.append(" ".join(current))
                current = []
        if current:
            paragraphs.append(" ".join(current))
    # Re-join hard-wrapped lines inside a paragraph.
    return [re.sub(r"\s*\n\s*", " ", p) for p in paragraphs]


def parse_pdf(filename: str, data: bytes) -> ParsedDocument:
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
    except Exception as exc:  # pypdf raises several exception types for corrupt files
        raise ParseError(f"Could not read PDF '{filename}': {exc}") from exc

    builder = _TextBuilder()
    pdf_title = None
    if reader.metadata and reader.metadata.title:
        pdf_title = str(reader.metadata.title).strip() or None
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = normalize_text(page.extract_text() or "")
        for i, para in enumerate(_pdf_page_paragraphs(page_text)):
            builder.add(para, {"page": page_number}, breaks_before=(i == 0 and page_number > 1))
    return ParsedDocument(
        builder.text,
        builder.blocks,
        "pdf",
        pdf_title or _default_title(filename),
        {"page_count": len(reader.pages)},
    )


# -------------------------------------------------------------------------- JSON
def _render_value(value: Any) -> str:
    if isinstance(value, list):
        if all(not isinstance(v, (dict, list)) for v in value):
            return ", ".join(str(v) for v in value)
        return "; ".join(_render_value(v) for v in value)
    if isinstance(value, dict):
        return "; ".join(f"{k}: {_render_value(v)}" for k, v in value.items())
    return str(value)


def _render_record(record: dict[str, Any], prefix: str = "") -> list[str]:
    lines: list[str] = []
    for key, value in record.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            lines.extend(_render_record(value, prefix=f"{path}."))
        else:
            lines.append(f"{path}: {_render_value(value)}")
    return lines


def _find_record_list(obj: Any, path: str = "") -> tuple[str, list[dict[str, Any]]] | None:
    """Find the largest list of objects in a JSON tree (the "records")."""
    best: tuple[str, list[dict[str, Any]]] | None = None
    if isinstance(obj, list) and obj and all(isinstance(x, dict) for x in obj):
        best = (path, obj)
    if isinstance(obj, dict):
        for key, value in obj.items():
            found = _find_record_list(value, f"{path}/{key}")
            if found and (best is None or len(found[1]) > len(best[1])):
                best = found
    return best


def parse_json(filename: str, data: bytes) -> ParsedDocument:
    try:
        obj = json.loads(_decode(data))
    except json.JSONDecodeError as exc:
        raise ParseError(f"Invalid JSON in '{filename}': {exc}") from exc

    builder = _TextBuilder()
    title = _default_title(filename)
    found = _find_record_list(obj)

    if isinstance(obj, dict):
        title = str(obj.get("title") or obj.get("name") or title)
        # Top-level scalar fields form a header block describing the collection.
        header = {
            k: v for k, v in obj.items() if not isinstance(v, (list, dict)) and k != "title"
        }
        if header:
            builder.add(
                normalize_text("\n".join(_render_record(header))),
                {"json_path": "/", "kind": "header"},
            )

    if found is not None:
        list_path, records = found
        for i, record in enumerate(records):
            builder.add(
                normalize_text("\n".join(_render_record(record))),
                {"json_path": f"{list_path}/{i}", "record_index": i, "kind": "record"},
                breaks_before=True,
            )
    elif isinstance(obj, dict):
        nested = {k: v for k, v in obj.items() if isinstance(v, (list, dict))}
        if nested:
            builder.add(
                normalize_text("\n".join(_render_record(nested))), {"json_path": "/", "kind": "body"}
            )
    else:
        builder.add(normalize_text(_render_value(obj)), {"json_path": "/", "kind": "body"})

    return ParsedDocument(builder.text, builder.blocks, "json", title)


_PARSERS = {
    "txt": parse_txt,
    "markdown": parse_markdown,
    "pdf": parse_pdf,
    "json": parse_json,
}
