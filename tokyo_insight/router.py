"""Query → candidate records, using the shipped routing pack (facts + vectors).

Coarse stage of the on-demand pipeline: pick the few records most likely to hold
the answer WITHOUT fetching any minutes. Fine-grained passage selection happens
later, locally, over only the fetched records (see live.py).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional

import numpy as np

from . import config


@dataclass
class Candidate:
    committee: str
    record: str
    url: str
    date: Optional[str]
    session: str
    speakers: List[str]
    score: float


@lru_cache(maxsize=1)
def _load():
    vecs = np.load(config.ROUTING_DIR / "routing_vectors.npy")
    pack = [json.loads(l) for l in
            (config.ROUTING_DIR / "routing_pack.jsonl")
            .read_text(encoding="utf-8").splitlines()]
    return vecs, pack


def route(query: str, model, k: Optional[int] = None,
          committee: Optional[str] = None) -> List[Candidate]:
    """Return up to k candidate records ranked by routing-vector similarity.

    Optionally restrict to one committee (a cheap, exact recall booster when the
    user already knows the arm). Never returns more than LIVE_MAX_FETCH.
    """
    k = min(k or config.LIVE_TOP_K, config.LIVE_MAX_FETCH)
    vecs, pack = _load()
    qv = model.encode([f"query: {query}"], normalize_embeddings=True,
                      show_progress_bar=False, convert_to_numpy=True).astype(np.float32)[0]
    sims = vecs @ qv
    order = np.argsort(-sims)
    out: List[Candidate] = []
    for i in order:
        p = pack[i]
        if committee and p["committee"] != committee:
            continue
        out.append(Candidate(p["committee"], p["record"], p["url"], p.get("date"),
                             p.get("session", ""), p.get("speakers", []),
                             float(sims[i])))
        if len(out) >= k:
            break
    return out
