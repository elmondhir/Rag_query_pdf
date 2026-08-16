import os
from typing import Any

import httpx
from dotenv import load_dotenv


load_dotenv()


OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
).rstrip("/")
OPENROUTER_CHAT_MODEL = os.getenv("OPENROUTER_CHAT_MODEL", "openrouter/free")
OPENROUTER_EMBED_MODEL = os.getenv(
    "OPENROUTER_EMBED_MODEL", "nvidia/nemotron-3-embed-1b:free"
)
OPENROUTER_EMBED_DIM = int(os.getenv("OPENROUTER_EMBED_DIM", "2048"))


def get_api_key() -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    return api_key


def _headers() -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {get_api_key()}",
        "Content-Type": "application/json",
    }
    if referer := os.getenv("OPENROUTER_HTTP_REFERER"):
        headers["HTTP-Referer"] = referer
    if app_name := os.getenv("OPENROUTER_APP_NAME"):
        headers["X-Title"] = app_name
    return headers


def embed_texts(texts: list[str], *, input_type: str) -> list[list[float]]:
    if not texts:
        return []
    if input_type not in {"query", "passage"}:
        raise ValueError("input_type must be 'query' or 'passage'")

    response = httpx.post(
        f"{OPENROUTER_BASE_URL}/embeddings",
        headers=_headers(),
        json={
            "model": OPENROUTER_EMBED_MODEL,
            "input": texts,
            "input_type": input_type,
            "encoding_format": "float",
        },
        timeout=60.0,
    )
    response.raise_for_status()
    data: list[dict[str, Any]] = response.json()["data"]
    data.sort(key=lambda item: item["index"])
    embeddings = [item["embedding"] for item in data]

    if len(embeddings) != len(texts):
        raise RuntimeError(
            f"OpenRouter returned {len(embeddings)} embeddings for {len(texts)} inputs"
        )
    if any(len(embedding) != OPENROUTER_EMBED_DIM for embedding in embeddings):
        raise RuntimeError(
            "Embedding size does not match OPENROUTER_EMBED_DIM "
            f"({OPENROUTER_EMBED_DIM})"
        )
    return embeddings


def chat_completion(messages: list[dict[str, str]]) -> str:
    response = httpx.post(
        f"{OPENROUTER_BASE_URL}/chat/completions",
        headers=_headers(),
        json={
            "model": OPENROUTER_CHAT_MODEL,
            "max_tokens": 1024,
            "temperature": 0.2,
            "messages": messages,
        },
        timeout=120.0,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("OpenRouter returned an empty chat response")
    return content.strip()
