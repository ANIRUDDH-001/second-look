"""A small, bounded client for a local Ollama server. Nothing here talks to the internet."""
from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass

import httpx

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("SECONDLOOK_MODEL", "gemma4:e2b-it-qat")
TIMEOUT_S = float(os.environ.get("SECONDLOOK_LLM_TIMEOUT", "240"))


class LLMError(RuntimeError):
    pass


@dataclass
class Reply:
    text: str
    seconds: float
    prompt_tokens: int | None
    output_tokens: int | None


def chat(messages: list[dict], schema: dict | None = None, images: list[bytes] | None = None,
         temperature: float = 0.0, num_predict: int = 1200) -> Reply:
    """One /api/chat call. Images attach to the last user message. Raises LLMError on any failure."""
    msgs = [dict(m) for m in messages]
    if images:
        msgs[-1]["images"] = [base64.b64encode(b).decode() for b in images]
    body = {"model": MODEL, "messages": msgs, "stream": False, "think": False,
            "options": {"temperature": temperature, "seed": 0, "num_predict": num_predict, "num_ctx": 8192}}
    if schema is not None:
        body["format"] = schema
    t0 = time.perf_counter()
    try:
        r = httpx.post(f"{OLLAMA_URL}/api/chat", json=body, timeout=TIMEOUT_S)
        r.raise_for_status()
        data = r.json()
    except (httpx.HTTPError, json.JSONDecodeError) as e:
        raise LLMError(f"ollama call failed: {e}") from e
    return Reply(data.get("message", {}).get("content", ""), time.perf_counter() - t0,
                 data.get("prompt_eval_count"), data.get("eval_count"))


def available() -> bool:
    try:
        r = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        return any(m.get("name", "").startswith(MODEL) for m in r.json().get("models", []))
    except httpx.HTTPError:
        return False
