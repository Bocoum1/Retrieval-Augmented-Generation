import requests
from rag.config import settings


def generate_answer(prompt: str) -> str:
    response = requests.post(
        settings.ollama_url,
        json={
            "model": settings.ollama_model,
            "prompt": prompt,
            "stream": False
        },
        timeout=180
    )
    response.raise_for_status()
    return response.json().get("response", "").strip()


def stream_answer(prompt: str):
    response = requests.post(
        settings.ollama_url,
        json={
            "model": settings.ollama_model,
            "prompt": prompt,
            "stream": True
        },
        timeout=180,
        stream=True
    )
    response.raise_for_status()

    for line in response.iter_lines():
        if not line:
            continue
        try:
            import json
            data = json.loads(line.decode("utf-8"))
            token = data.get("response", "")
            if token:
                yield token
        except Exception:
            continue