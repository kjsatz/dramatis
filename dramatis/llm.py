"""Thin client for the story generator (Gemini / Gemma via the google-genai SDK).

Credentials come from .env (python-dotenv): GEMINI_API_KEY for AI Studio, or
GOOGLE_GENAI_USE_VERTEXAI=true + GOOGLE_CLOUD_PROJECT + GOOGLE_CLOUD_LOCATION for Vertex; the SDK
reads those itself. Nothing here spends money on import; `generate` does.
"""

import random
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

RETRYABLE = ("429", "500", "502", "503", "504", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "DEADLINE")


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    finish_reason: str | None
    seconds: float


class LLM:
    def __init__(self, model: str, temperature: float = 1.0, max_output_tokens: int = 1024):
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.client = genai.Client()

    def generate(self, prompt: str, retries: int = 6) -> Completion:
        """One completion, with exponential backoff on rate limits and server errors."""
        config = types.GenerateContentConfig(
            temperature=self.temperature, max_output_tokens=self.max_output_tokens
        )
        for attempt in range(retries + 1):
            t0 = time.perf_counter()
            try:
                r = self.client.models.generate_content(
                    model=self.model, contents=prompt, config=config
                )
            except Exception as e:  # the SDK raises several types; match on the message
                if attempt < retries and any(k in str(e) for k in RETRYABLE):
                    time.sleep(min(60, 2**attempt) * (0.5 + random.random()))
                    continue
                raise
            usage = r.usage_metadata
            finish = None
            if r.candidates:
                finish = getattr(r.candidates[0].finish_reason, "name", None)
            return Completion(
                text=r.text or "",
                model=self.model,
                input_tokens=getattr(usage, "prompt_token_count", None),
                output_tokens=getattr(usage, "candidates_token_count", None),
                finish_reason=finish,
                seconds=time.perf_counter() - t0,
            )
        raise RuntimeError("unreachable")
