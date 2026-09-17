"""Thin client for the story generator, with two backends behind one interface.

- "vertex": Gemini / Gemma via the google-genai SDK. Credentials from .env (python-dotenv):
  GEMINI_API_KEY for AI Studio, or GOOGLE_GENAI_USE_VERTEXAI=true + GOOGLE_CLOUD_PROJECT +
  GOOGLE_CLOUD_LOCATION for Vertex; the SDK reads those itself.
- "openai": any OpenAI-compatible server, e.g. vLLM serving the same Gemma weights on a rented
  GPU (`vllm serve google/gemma-4-26B-A4B-it`), reached at `base_url`.

Nothing here spends money on import; `generate` does.
"""

import os
import random
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

RETRYABLE = ("429", "500", "502", "503", "504", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "DEADLINE",
             "imeout", "imed out", "Connect", "nodename")  # fmt: skip
# ("Connect": httpx ConnectError and dropped connections; "nodename": macOS DNS hiccups.)
BACKPRESSURE = ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE")  # retried without limit


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    finish_reason: str | None
    seconds: float


class LLM:
    def __init__(
        self,
        model: str,
        temperature: float = 1.0,
        max_output_tokens: int = 1024,
        json_schema: dict | None = None,
        thinking_budget: int | None = None,
        backend: str = "vertex",
        base_url: str | None = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.json_schema = json_schema  # if set, the model must return JSON matching it
        # Thinking models spend output tokens on reasoning before answering; 0 turns it off.
        self.thinking_budget = thinking_budget
        self.backend = backend
        if backend == "vertex":
            # A per-request timeout: without one, a connection the server silently drops blocks
            # its worker thread forever (bundle 2 stalled for 4 h this way). A story takes ~6 s;
            # queue stalls of ~2 min were seen, so 3 min is generous. Timeouts are retryable.
            self.client = genai.Client(http_options={"timeout": 180_000})
        elif backend == "openai":
            from openai import OpenAI  # local import: only needed for this backend

            # vLLM ignores the key; Vertex's OpenAI-compatible endpoint wants a gcloud access token.
            api_key = os.environ.get("OPENAI_API_KEY", "unused")
            self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=600)
        else:
            raise ValueError(f"unknown backend {backend!r}")
        self.n_retries = 0  # total retries across calls (backpressure indicator)

    def generate(self, prompt: str, retries: int = 6) -> Completion:
        """One completion. Rate limits / overload (429, 503) are treated as backpressure and
        retried without limit with capped jittered backoff; other retryable errors get `retries`
        attempts; anything else raises."""
        attempt = 0
        while True:
            t0 = time.perf_counter()
            try:
                return self._call(prompt, t0)
            except Exception as e:  # the SDKs raise several types; match on the message
                msg = str(e)
                backpressure = any(k in msg for k in BACKPRESSURE)
                if backpressure or (attempt < retries and any(k in msg for k in RETRYABLE)):
                    attempt += 1
                    self.n_retries += 1
                    time.sleep(min(30, 2 ** min(attempt, 5)) * (0.5 + random.random()))
                    continue
                raise

    def _call(self, prompt: str, t0: float) -> Completion:
        if self.backend == "openai":
            kwargs = {}
            if self.json_schema:
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "verdict", "schema": self.json_schema},
                }
            r = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
                **kwargs,
            )
            choice = r.choices[0]
            return Completion(
                text=choice.message.content or "",
                model=self.model,
                input_tokens=getattr(r.usage, "prompt_tokens", None),
                output_tokens=getattr(r.usage, "completion_tokens", None),
                finish_reason=choice.finish_reason,
                seconds=time.perf_counter() - t0,
            )

        config = types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            response_mime_type="application/json" if self.json_schema else None,
            response_json_schema=self.json_schema,
            thinking_config=(
                types.ThinkingConfig(thinking_budget=self.thinking_budget)
                if self.thinking_budget is not None
                else None
            ),
        )
        r = self.client.models.generate_content(model=self.model, contents=prompt, config=config)
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
