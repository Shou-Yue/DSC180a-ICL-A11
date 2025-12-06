import os
from typing import Optional

from openai import OpenAI, APITimeoutError


def _get_client(timeout: float = 10.0) -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment")
    # global timeout for httpx
    return OpenAI(api_key=api_key, timeout=timeout)


def _openai_predict(prompt: str, model_name: Optional[str] = None) -> Optional[float]:
    client = _get_client(timeout=10.0)  # shorten how long we wait

    model = model_name or os.getenv("GPT5_MODEL") or "gpt-4.1-mini"

    try:
        resp = client.responses.create(
            model=model,
            input=prompt,
        )
    except APITimeoutError:
        print("[WARN] OpenAI request timed out; skipping this task.")
        return None
    except Exception as e:
        print(f"[WARN] OpenAI request failed: {e!r}")
        return None

    # Depending on SDK version; this is the common pattern
    try:
        text = resp.output_text
    except AttributeError:
        text = resp.output[0].content[0].text

    try:
        return float(text.strip())
    except ValueError:
        print(f"[WARN] Could not parse float from model output: {text!r}")
        return None


def _mock_predict(prompt: str) -> float:
    # Fast deterministic placeholder for debugging. You can change this.
    return 0.0


def llm_predict(prompt: str) -> Optional[float]:
    backend = os.getenv("LLM_BACKEND", "openai")
    if backend.lower() == "mock":
        return _mock_predict(prompt)
    return _openai_predict(prompt)
