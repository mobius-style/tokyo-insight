"""Minimal OpenAI-compatible chat client (urllib; no SDK dependency).

120B-first but provider-agnostic: endpoint/model/key from env. BYO API key.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Dict

from . import config


@dataclass
class LLMResponse:
    ok: bool
    text: str = ""
    error: str = ""
    latency_ms: int = 0
    usage: Dict = field(default_factory=dict)


def _api_key() -> str:
    key = os.environ.get("TOKYO_INSIGHT_API_KEY") or os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("set TOKYO_INSIGHT_API_KEY or GROQ_API_KEY for the LLM endpoint")
    return key


def chat(system: str, user: str, *, max_tokens: int = 2400,
         temperature: float = 0.2) -> LLMResponse:
    body = json.dumps({
        "model": config.LLM_MODEL,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "max_tokens": max_tokens, "temperature": temperature,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{config.LLM_API_BASE}/chat/completions", data=body,
        headers={"Authorization": f"Bearer {_api_key()}",
                 "Content-Type": "application/json",
                 "Accept": "application/json",
                 # urllib's default UA is blocked by Cloudflare (403/1010);
                 # send a normal client UA.
                 "User-Agent": "tokyo-insight/0.1 (+https://mobius; python)"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read().decode("utf-8"))
        return LLMResponse(
            ok=True, text=data["choices"][0]["message"]["content"],
            latency_ms=int((time.time() - t0) * 1000), usage=data.get("usage", {}))
    except Exception as e:  # noqa: BLE001
        detail = ""
        if hasattr(e, "read"):
            try:
                detail = e.read().decode("utf-8", "replace")[:300]
            except Exception:
                pass
        return LLMResponse(ok=False, error=f"{e} {detail}".strip(),
                           latency_ms=int((time.time() - t0) * 1000))
