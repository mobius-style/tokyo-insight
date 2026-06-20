"""Query → candidate records via the shipped routing pack.

Agenda-section routing: the pack holds one vector per (record × agenda section);
a record is scored by its BEST-matching section (max-pool), which is far sharper
than a single record-level mean vector. The coarse stage of the on-demand
pipeline — pick the few records most likely to hold the answer WITHOUT fetching
any minutes; fine passage selection happens later, locally (see live.py).

Back-compatible: if section_records.npy is absent, each vector IS one record.
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
    sr_path = config.ROUTING_DIR / "section_records.npy"
    if sr_path.exists():
        sec_rec = np.load(sr_path)            # section row -> record index
    else:
        sec_rec = np.arange(len(pack))        # record-level pack (1 vec / record)
    return vecs, pack, sec_rec


def route(query: str, model, k: Optional[int] = None,
          committee: Optional[str] = None) -> List[Candidate]:
    """Up to k candidate records, ranked by best-matching section similarity.
    Optionally restrict to one committee. Never exceeds LIVE_MAX_FETCH."""
    k = min(k or config.LIVE_TOP_K, config.LIVE_MAX_FETCH)
    vecs, pack, sec_rec = _load()
    qv = model.encode([f"query: {query}"], normalize_embeddings=True,
                      show_progress_bar=False, convert_to_numpy=True).astype(np.float32)[0]
    sims = vecs @ qv
    order = np.argsort(-sims)                  # sections, best first
    out: List[Candidate] = []
    seen = set()
    for si in order:
        ri = int(sec_rec[si])
        if ri in seen:
            continue                           # record already taken (its best section)
        p = pack[ri]
        if committee and p["committee"] != committee:
            continue
        seen.add(ri)
        out.append(Candidate(p["committee"], p["record"], p["url"], p.get("date"),
                             p.get("session", ""), p.get("speakers", []),
                             float(sims[si])))
        if len(out) >= k:
            break
    return out
