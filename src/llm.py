from openai import OpenAI

from .config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
from .retriever import build_context


SYSTEM_PROMPT = """You are a research assistant that answers questions using only the supplied document evidence.

Rules:
1. Do not invent facts that are not supported by the evidence.
2. If the evidence is insufficient, clearly say that the uploaded documents do not provide enough information.
3. Cite evidence using [Source N] markers.
4. Prefer concise, precise answers.
5. When multiple sources disagree, describe the disagreement instead of silently choosing one.
"""


def generate_answer(question: str, hits):
    if not OPENAI_API_KEY:
        return (
            "LLM generation is not configured. Review the retrieved evidence below "
            "or add OPENAI_API_KEY to enable generated answers."
        )

    context = build_context(hits)
    client_kwargs = {"api_key": OPENAI_API_KEY}
    if OPENAI_BASE_URL:
        client_kwargs["base_url"] = OPENAI_BASE_URL

    client = OpenAI(**client_kwargs)

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.1,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Document evidence:\n{context}\n\n"
                    "Answer using only this evidence."
                ),
            },
        ],
    )
    return response.choices[0].message.content.strip()
