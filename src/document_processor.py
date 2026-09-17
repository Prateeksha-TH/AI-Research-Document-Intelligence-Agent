from dataclasses import dataclass
from pathlib import Path
import re
import fitz

from .config import CHUNK_OVERLAP, CHUNK_SIZE


@dataclass
class DocumentChunk:
    text: str
    source: str
    page: int
    chunk_id: str


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    text = clean_text(text)
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start + chunk_size // 2:
                end = boundary

        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)

        if end >= len(text):
            break

        start = max(end - overlap, start + 1)

    return chunks


def extract_pdf_chunks(pdf_path: str | Path):
    pdf_path = Path(pdf_path)
    results = []

    with fitz.open(pdf_path) as doc:
        for page_number, page in enumerate(doc, start=1):
            text = clean_text(page.get_text("text"))
            if not text:
                continue

            for index, chunk in enumerate(chunk_text(text)):
                results.append(
                    DocumentChunk(
                        text=chunk,
                        source=pdf_path.name,
                        page=page_number,
                        chunk_id=f"{pdf_path.name}:p{page_number}:c{index}",
                    )
                )

    return results
