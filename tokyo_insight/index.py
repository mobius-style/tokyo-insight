"""Local index: chunk → e5 embed → FAISS(IP) + inline BM25, persisted on disk.

Stores ONLY locally-derived data on the user's machine (embeddings + chunk
metadata). The package never ships minutes text.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import config
from .parse import Meeting, parse_html


@dataclass
class Chunk:
    chunk_id: str
    text: str
    committee: str
    record: str
    url: str
    meeting_kind: Optional[str]
    meeting_date: Optional[str]
    agenda_item: Optional[str]
    speaker_name: Optional[str]
    speaker_role: Optional[str]


def char_bigrams(text: str) -> List[str]:
    cleaned = re.sub(r"\s+", "", text or "")
    out, i, n = [], 0, len(cleaned)
    while i < n:
        c = cleaned[i]
        if c.isascii() and (c.isalnum() or c in "-_"):
            j = i
            while j < n and cleaned[j].isascii() and (cleaned[j].isalnum() or cleaned[j] in "-_"):
                j += 1
            out.append(cleaned[i:j].lower())
            i = j
        else:
            out.append(cleaned[i:i + 2] if i + 1 < n else cleaned[i])
            i += 1
    return out


class BM25Okapi:
    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b, self.N = k1, b, len(corpus)
        self.doc_len = [len(d) for d in corpus]
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 0.0
        self.postings: Dict[str, List[Tuple[int, int]]] = {}
        df: Dict[str, int] = {}
        for i, d in enumerate(corpus):
            f: Dict[str, int] = {}
            for t in d:
                f[t] = f.get(t, 0) + 1
            for t, c in f.items():
                self.postings.setdefault(t, []).append((i, c))
                df[t] = df.get(t, 0) + 1
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def get_scores(self, query: List[str]) -> np.ndarray:
        s = np.zeros(self.N)
        for t in query:
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, f in self.postings[t]:
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                s[i] += idf * f * (self.k1 + 1) / denom
        return s


def chunk_meeting(m: Meeting) -> List[Chunk]:
    chunks: List[Chunk] = []
    for idx, u in enumerate(m.utterances):
        text = (u.text or "").strip()
        if not text:
            continue
        pieces = ([text] if len(text) <= config.MAX_CHUNK_CHARS
                  else [text[i:i + config.MAX_CHUNK_CHARS]
                        for i in range(0, len(text), config.MAX_CHUNK_CHARS)])
        for pi, p in enumerate(pieces):
            chunks.append(Chunk(
                chunk_id=f"{m.committee_slug}_{m.record}_u{idx:04d}_p{pi:02d}",
                text=p, committee=m.committee_slug, record=m.record, url=m.url,
                meeting_kind=m.meeting_kind, meeting_date=m.meeting_date,
                agenda_item=u.agenda_item, speaker_name=u.speaker_name,
                speaker_role=u.speaker_role))
    return chunks


def _pick_device() -> str:
    """GPU when free, else CPU. Override with TOKYO_INSIGHT_DEVICE=cpu|cuda."""
    import os
    forced = os.environ.get("TOKYO_INSIGHT_DEVICE")
    if forced:
        return forced
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _embedder():
    import os
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(config.E5_MODEL_ID, device=_pick_device())


def build() -> Dict:
    """Parse all fetched raw HTML, chunk, embed, persist the index."""
    raw_files = sorted(config.RAW_DIR.rglob("*.html"))
    if not raw_files:
        raise FileNotFoundError(f"no fetched records under {config.RAW_DIR}; run `fetch` first")
    all_chunks: List[Chunk] = []
    for p in raw_files:
        slug, rec = p.parent.name, p.stem
        url = f"{config.BASE_URL}/{slug}/{rec}.html"
        m = parse_html(p.read_text(encoding="utf-8"), slug=slug, rec=rec, url=url)
        all_chunks.extend(chunk_meeting(m))
    if not all_chunks:
        raise ValueError("no utterances parsed from fetched records")

    import os
    model = _embedder()
    on_gpu = getattr(model, "device", None) and "cuda" in str(model.device)
    batch_size = int(os.environ.get("TOKYO_INSIGHT_BATCH", 256 if on_gpu else 8))
    emb = model.encode([f"passage: {c.text}" for c in all_chunks],
                       normalize_embeddings=True, batch_size=batch_size,
                       show_progress_bar=bool(os.environ.get("TOKYO_INSIGHT_PROGRESS")),
                       convert_to_numpy=True).astype(np.float32)

    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(config.INDEX_DIR / "embeddings.npy", emb)
    with (config.INDEX_DIR / "chunks.jsonl").open("w", encoding="utf-8") as fh:
        for c in all_chunks:
            fh.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    meta = {"chunks": len(all_chunks), "records": len(raw_files),
            "embedding_model": config.E5_MODEL_ID, "dim": int(emb.shape[1])}
    (config.INDEX_DIR / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def load() -> Tuple[List[Chunk], "object", BM25Okapi]:
    import faiss
    emb = np.load(config.INDEX_DIR / "embeddings.npy")
    chunks = [Chunk(**json.loads(l))
              for l in (config.INDEX_DIR / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    bm25 = BM25Okapi([char_bigrams(c.text) for c in chunks])
    return chunks, index, bm25
