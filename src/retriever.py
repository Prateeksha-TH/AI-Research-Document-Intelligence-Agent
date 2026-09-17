from .config import DEFAULT_TOP_K


def retrieve(store, question: str, top_k: int = DEFAULT_TOP_K):
    if not question or not question.strip():
        return []
    return store.search(question.strip(), top_k=top_k)


def build_context(hits):
    blocks = []
    for i, hit in enumerate(hits, start=1):
        blocks.append(
            f"[Source {i}] {hit['source']} — page {hit['page']}\n"
            f"{hit['text']}"
        )
    return "\n\n".join(blocks)
