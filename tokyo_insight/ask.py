"""Retrieve → governance prompt → LLM → citation-grounded answer."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from . import config, governance, llm
from .index import Chunk, BM25Okapi, char_bigrams, load


def _minmax(s: np.ndarray) -> np.ndarray:
    if s.size == 0:
        return s
    lo, hi = float(s.min()), float(s.max())
    return np.zeros_like(s) if hi - lo < 1e-9 else (s - lo) / (hi - lo)


def _retrieve(query: str, model, index, bm25: BM25Okapi,
              chunks: List[Chunk]) -> List[int]:
    q = model.encode([f"query: {query}"], normalize_embeddings=True,
                     show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
    ds, di = index.search(q, k=config.TOP_K_DENSE)
    ds, di = ds[0], di[0]
    bm = bm25.get_scores(char_bigrams(query))
    cand: Dict[int, Dict[str, float]] = {}
    for s, i in zip(ds, di):
        if i >= 0:
            cand[int(i)] = {"d": float(s), "b": float(bm[i])}
    for i in np.argsort(-bm)[:config.TOP_K_BM25]:
        cand.setdefault(int(i), {"d": 0.0, "b": float(bm[i])})
    ci = list(cand)
    dn = _minmax(np.array([cand[i]["d"] for i in ci]))
    bn = _minmax(np.array([cand[i]["b"] for i in ci]))
    hy = config.DENSE_WEIGHT * dn + config.BM25_WEIGHT * bn
    return [ci[i] for i in np.argsort(-hy)[:config.HYBRID_TOPN]]


def _format(ci: List[Tuple[int, Chunk]]) -> str:
    out = []
    for n, (_, c) in enumerate(ci, 1):
        out.append(f"[{n}] source_kind=council_minutes  会議={c.meeting_kind} / {c.meeting_date}  "
                   f"議題={c.agenda_item or '議題未特定'}  発言者={c.speaker_name} ({c.speaker_role})  "
                   f"url={c.url}")
        out.append(c.text)
        out.append("")
    return "\n".join(out)


def ask(question: str) -> Dict:
    from .index import _embedder
    chunks, index, bm25 = load()
    model = _embedder()
    idxs = _retrieve(question, model, index, bm25, chunks)
    ci = [(i, chunks[i]) for i in idxs]
    user = "## Retrieved sources\n\n" + _format(ci) + "\n## Question\n\n" + question
    resp = llm.chat(governance.SYSTEM, user)
    return {
        "question": question,
        "answer": resp.text if resp.ok else f"[LLM error] {resp.error}",
        "ok": resp.ok,
        "latency_ms": resp.latency_ms,
        "usage": resp.usage,
        "retrieved": [{"n": n, "chunk_id": c.chunk_id, "speaker": c.speaker_name,
                       "role": c.speaker_role, "agenda": c.agenda_item,
                       "meeting": f"{c.meeting_kind}/{c.meeting_date}", "url": c.url}
                      for n, (_, c) in enumerate(ci, 1)],
    }
