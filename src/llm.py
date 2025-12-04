import os
from typing import Optional

import openai

GPT5_MODEL = os.environ.get("GPT5_MODEL", "gpt-5")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise EnvironmentError(
        "OPENAI_API_KEY not set; export it in your environment before running."
    )

openai.api_key = OPENAI_API_KEY


def openai_predict(prompt: str, model_name: Optional[str] = None) -> float:
    """
    GPT API inference. Returns a float parsed from the model's text output.
    This matches the usage in your notebook: the model is expected to output
    a bare float as text.
    """
    if model_name is None:
        model_name = GPT5_MODEL
    client = openai.OpenAI()
    response = client.responses.create(
        model=model_name,
        input=prompt,
    )
    return float(response.output_text)


def llm_predict(prompt: str) -> float:
    """
    Route to the active backend. Currently only the OpenAI backend is wired up,
    mirroring the original notebook.
    """
    return openai_predict(prompt)
