import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import docx
import pymupdf

from config import CHUNK_OVERLAP, CHUNK_SIZE

SUPPORTED_EXTENSIONS = ["pdf", "docx", "txt"]


def _split_text_into_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """Recursively splits text by paragraphs, sentences, or characters trying to respect chunk_size."""
    text = text.strip()
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    paragraphs = text.split("\n\n")
    chunks: List[str] = []
    current_chunk: List[str] = []
    current_len = 0

    for para in paragraphs:
        para_trimmed = para.strip()
        if not para_trimmed:
            continue

        if len(para_trimmed) > chunk_size:
            # Paragraph itself is too large, split by single lines or sentences
            sub_lines = para_trimmed.replace(". ", ".\n").split("\n")
            for line in sub_lines:
                line_trimmed = line.strip()
                if not line_trimmed:
                    continue

                if current_len + len(line_trimmed) + 1 > chunk_size and current_chunk:
                    chunk_text = "\n".join(current_chunk).strip()
                    if chunk_text:
                        chunks.append(chunk_text)
                    # Keep overlap
                    overlap_len = 0
                    new_current = []
                    for prev_item in reversed(current_chunk):
                        if overlap_len + len(prev_item) <= chunk_overlap:
                            new_current.insert(0, prev_item)
                            overlap_len += len(prev_item) + 1
                        else:
                            break
                    current_chunk = new_current
                    current_len = sum(len(x) + 1 for x in current_chunk)

                current_chunk.append(line_trimmed)
                current_len += len(line_trimmed) + 1
        else:
            if current_len + len(para_trimmed) + 1 > chunk_size and current_chunk:
                chunk_text = "\n\n".join(current_chunk).strip()
                if chunk_text:
                    chunks.append(chunk_text)
                # Keep overlap
                overlap_len = 0
                new_current = []
                for prev_item in reversed(current_chunk):
                    if overlap_len + len(prev_item) <= chunk_overlap:
                        new_current.insert(0, prev_item)
                        overlap_len += len(prev_item) + 1
                    else:
                        break
                current_chunk = new_current
                current_len = sum(len(x) + 1 for x in current_chunk)

            current_chunk.append(para_trimmed)
            current_len += len(para_trimmed) + 2

    if current_chunk:
        chunk_text = "\n\n".join(current_chunk).strip()
        if chunk_text:
            chunks.append(chunk_text)

    return chunks


def _extract_pdf(path: Path) -> List[Dict]:
    """Extracts text from PDF page by page."""
    chunks = []
    chunk_counter = 0

    try:
        doc = pymupdf.open(path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text").strip()
            if not text:
                continue

            page_chunks = _split_text_into_chunks(text)
            for chunk in page_chunks:
                chunk_counter += 1
                unique_id = f"{path.name}_p{page_num + 1}_c{chunk_counter}_{uuid.uuid4().hex[:6]}"
                chunks.append(
                    {
                        "document": chunk,
                        "metadata": {
                            "source": path.name,
                            "page": page_num + 1,
                            "chunk_index": chunk_counter,
                            "chunk_id": unique_id,
                            "created_at": datetime.now().isoformat(),
                        },
                    }
                )
        doc.close()
    except Exception as exc:
        print(f"Error processing PDF {path}: {exc}")

    return chunks


def _extract_docx(path: Path) -> List[Dict]:
    """Extracts text from DOCX file."""
    chunks = []
    try:
        doc = docx.Document(path)
        paragraphs = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
        full_text = "\n\n".join(paragraphs)

        text_chunks = _split_text_into_chunks(full_text)
        for i, chunk in enumerate(text_chunks, start=1):
            unique_id = f"{path.name}_c{i}_{uuid.uuid4().hex[:6]}"
            chunks.append(
                {
                    "document": chunk,
                    "metadata": {
                        "source": path.name,
                        "page": 1,
                        "chunk_index": i,
                        "chunk_id": unique_id,
                        "created_at": datetime.now().isoformat(),
                    },
                }
            )
    except Exception as exc:
        print(f"Error processing DOCX {path}: {exc}")

    return chunks


def _extract_txt(path: Path) -> List[Dict]:
    """Extracts text from UTF-8 text file."""
    chunks = []
    try:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            return []

        text_chunks = _split_text_into_chunks(text)
        for i, chunk in enumerate(text_chunks, start=1):
            unique_id = f"{path.name}_c{i}_{uuid.uuid4().hex[:6]}"
            chunks.append(
                {
                    "document": chunk,
                    "metadata": {
                        "source": path.name,
                        "page": 1,
                        "chunk_index": i,
                        "chunk_id": unique_id,
                        "created_at": datetime.now().isoformat(),
                    },
                }
            )
    except Exception as exc:
        print(f"Error processing TXT {path}: {exc}")

    return chunks


def extract_texts_from_file(path: Path) -> List[Dict]:
    """Extracts text and splits into chunks with metadata for supported document types."""
    if isinstance(path, str):
        path = Path(path)

    if not path.exists():
        return []

    suffix = path.suffix.lower().lstrip(".")
    if suffix == "pdf":
        return _extract_pdf(path)
    if suffix == "docx":
        return _extract_docx(path)
    if suffix == "txt":
        return _extract_txt(path)

    return []

