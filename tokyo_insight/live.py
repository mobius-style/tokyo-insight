"""On-demand (live) ask: route → fetch only the candidate records → retrieve
within them locally → cited answer. Stores no corpus; fetched HTML is a
session-local cache (config.RAW_DIR) reused on repeat questions.

Two-stage retrieval: coarse record routing (router.route, no minutes fetched),
then fine chunk-level hybrid retrieval over ONLY the fetched records. The fetch
cap (config.LIVE_MAX_FETCH) is the anti-bulk-crawl / politeness guard.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np

from . import config, governance, llm
from .ask import _format, _minmax           # reuse identical formatting / scaling
from .fetch import fetch_record
from .index import BM25Okapi, Chunk, char_bigrams, chunk_meeting, _embedder
from .parse import parse_html
from .router import Candidate, route


def _ensure_local(c: Candidate) -> bool:
    """Cache-first fetch. Returns True if a network fetch happened."""
    path = config.RAW_DIR / c.committee / f"{c.record}.html"
    if path.exists() and path.stat().st_size > 0:
        return False
    fetch_record(c.committee, c.record)      # robots-guarded inside
    return True


def _gather(cands: List[Candidate]) -> List[Chunk]:
    chunks: List[Chunk] = []
    fetched = 0
    for c in cands:
        if _ensure_local(c):
            fetched += 1
            if fetched < len(cands):
                time.sleep(config.REQUEST_DELAY_SEC)   # polite between net fetches
        path = config.RAW_DIR / c.committee / f"{c.record}.html"
        m = parse_html(path.read_text(encoding="utf-8"),
                       slug=c.committee, rec=c.record, url=c.url)
        chunks.extend(chunk_meeting(m))
    return chunks


def _retrieve_local(query: str, model, chunks: List[Chunk]) -> List[int]:
    if not chunks:
        return []
    emb = model.encode([f"passage: {c.text}" for c in chunks],
                       normalize_embeddings=True, show_progress_bar=False,
                       convert_to_numpy=True).astype(np.float32)
    qv = model.encode([f"query: {query}"], normalize_embeddings=True,
                      show_progress_bar=False, convert_to_numpy=True).astype(np.float32)[0]
    dense = emb @ qv
    bm25 = BM25Okapi([char_bigrams(c.text) for c in chunks])
    bm = bm25.get_scores(char_bigrams(query))
    hybrid = config.DENSE_WEIGHT * _minmax(dense) + config.BM25_WEIGHT * _minmax(bm)
    return list(np.argsort(-hybrid)[:config.HYBRID_TOPN])


def ask_live(question: str, committee: Optional[str] = None,
             answer: bool = True) -> Dict:
    model = _embedder()
    cands = route(question, model, committee=committee)
    chunks = _gather(cands)
    idxs = _retrieve_local(question, model, chunks)
    ci = [(i, chunks[i]) for i in idxs]
    routed = [{"committee": c.committee, "record": c.record, "session": c.session,
               "score": round(c.score, 3), "url": c.url} for c in cands]
    out: Dict = {
        "question": question,
        "routed_records": routed,
        "retrieved": [{"n": n, "chunk_id": c.chunk_id, "speaker": c.speaker_name,
                       "role": c.speaker_role, "agenda": c.agenda_item,
                       "meeting": f"{c.meeting_kind}/{c.meeting_date}", "url": c.url}
                      for n, (_, c) in enumerate(ci, 1)],
    }
    if not answer:                 # offline mode: routing + retrieval only, no key
        return out
    user = "## Retrieved sources\n\n" + _format(ci) + "\n## Question\n\n" + question
    resp = llm.chat(governance.SYSTEM, user)
    out.update(answer=resp.text if resp.ok else f"[LLM error] {resp.error}",
               ok=resp.ok, latency_ms=resp.latency_ms, usage=resp.usage)
    return out
