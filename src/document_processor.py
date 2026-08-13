import fitz
import docx
from pathlib import Path
from typing import List, Dict


def extract_pdf_diagrams(path: Path, max_images: int = 6) -> List[Dict]:
    if not path.exists() or path.suffix.lower() != ".pdf":
        return []

    diagrams: List[Dict] = []
    doc = fitz.open(path)
    total_pages = min(len(doc), 6)

    for page_num in range(total_pages):
        page = doc.load_page(page_num)
        image_refs = page.get_images(full=True)
        if image_refs:
            for img_index, img in enumerate(image_refs[:max(1, max_images // max(1, total_pages))]):
                try:
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    if pix.n_channels == 4:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    image_bytes = pix.tobytes("png")
                    diagrams.append(
                        {
                            "source": path.name,
                            "page": page_num + 1,
                            "image_index": img_index + 1,
                            "image_bytes": image_bytes,
                        }
                    )
                    pix = None
                except Exception:
                    continue
                if len(diagrams) >= max_images:
                    return diagrams

        if not image_refs or len(diagrams) < max_images:
            try:
                page_pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
                diagrams.append(
                    {
                        "source": path.name,
                        "page": page_num + 1,
                        "image_index": 1,
                        "image_bytes": page_pix.tobytes("png"),
                    }
                )
            except Exception:
                continue

        if len(diagrams) >= max_images:
            break

    return diagrams

SUPPORTED_EXTENSIONS = ["pdf", "docx", "txt"]
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def _split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    sentences = text.replace("\n", " ").split(" ")
    chunks = []
    current = []
    current_len = 0
    for token in sentences:
        if not token:
            continue
        current.append(token)
        current_len += len(token) + 1
        if current_len >= chunk_size:
            chunks.append(" ".join(current).strip())
            current = current[-overlap // 5 :]
            current_len = sum(len(t) + 1 for t in current)
    if current:
        chunks.append(" ".join(current).strip())
    return chunks


def _extract_pdf(path: Path) -> List[Dict]:
    doc = fitz.open(path)
    chunks = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text().strip()
        if not text:
            continue
        page_chunks = _split_text(text)
        for chunk in page_chunks:
            chunks.append({"document": chunk, "metadata": {"source": path.name, "page": page_num + 1}})
    return chunks


def _extract_docx(path: Path) -> List[Dict]:
    doc = docx.Document(path)
    text = []
    for para in doc.paragraphs:
        if para.text.strip():
            text.append(para.text.strip())
    joined = "\n".join(text)
    chunks = []
    for chunk in _split_text(joined):
        chunks.append({"document": chunk, "metadata": {"source": path.name}})
    return chunks


def _extract_txt(path: Path) -> List[Dict]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    chunks = []
    for chunk in _split_text(text):
        chunks.append({"document": chunk, "metadata": {"source": path.name}})
    return chunks


def extract_texts_from_file(path: Path) -> List[Dict]:
    suffix = path.suffix.lower().strip(".")
    if suffix == "pdf":
        return _extract_pdf(path)
    if suffix == "docx":
        return _extract_docx(path)
    if suffix == "txt":
        return _extract_txt(path)
    return []
