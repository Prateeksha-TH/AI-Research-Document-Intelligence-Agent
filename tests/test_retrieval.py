from src.retriever import build_context


def test_build_context_contains_source_metadata():
    hits = [{
        "text": "The policy was introduced in 2025.",
        "source": "policy.pdf",
        "page": 4,
        "chunk_id": "x",
        "distance": 0.12,
    }]

    context = build_context(hits)

    assert "[Source 1]" in context
    assert "policy.pdf" in context
    assert "page 4" in context
    assert "introduced in 2025" in context
