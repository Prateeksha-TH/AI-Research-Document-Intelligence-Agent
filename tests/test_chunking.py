from src.document_processor import chunk_text


def test_chunk_text_returns_multiple_overlapping_chunks():
    text = "word " * 1000
    chunks = chunk_text(text, chunk_size=100, overlap=20)

    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)


def test_chunk_text_empty_input():
    assert chunk_text("") == []
    assert chunk_text("   ") == []
