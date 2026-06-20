"""Incremental, idempotent routing-pack refresh (user-initiated).

Diffs the robots-permitted landing pages (facts) against the local routing pack,
fetches ONLY the new records, embeds + mean-pools them, and appends to the pack.
Never re-crawls the whole site; never stores minutes text in the pack. Trigger is
the user running `refresh` — there is no silent background daemon.
"""
from __future__ import annotations

import datetime
import json
import re
import time
from typing import List, Optional, Tuple

import numpy as np

from . import config
from .fetch import _guard, fetch_record, list_records
from .index import _embedder, chunk_meeting
from .parse import parse_html

_ZEN = str.maketrans("０１２３４５６７８９", "0123456789")
_ERA = {"令和": 2018, "平成": 1988, "昭和": 1925}   # 令和N = 2018+N, etc.


def _jp_date(label: str) -> Optional[str]:
    """'第14号（令和7年11月18日）' -> '2025-11-18' (facts from the landing label)."""
    m = re.search(r"(令和|平成|昭和)(\d+|元)年(\d+)月(\d+)日", label.translate(_ZEN))
    if not m:
        return None
    era, y, mo, d = m.groups()
    yy = 1 if y == "元" else int(y)
    return f"{_ERA[era] + yy:04d}-{int(mo):02d}-{int(d):02d}"


def _session(record: str) -> str:
    m = re.match(r"\d{4}-(\d+)", record)
    return f"第{int(m.group(1))}号" if m else ""


def _load_pack():
    vfile = config.ROUTING_DIR / "routing_vectors.npy"
    pfile = config.ROUTING_DIR / "routing_pack.jsonl"
    if vfile.exists() and pfile.exists():
        vecs = np.load(vfile)
        pack = [json.loads(l) for l in
                pfile.read_text(encoding="utf-8").splitlines()]
        return vecs, pack
    return np.zeros((0, 0), dtype=np.float32), []


def pack_meta() -> dict:
    f = config.ROUTING_DIR / "meta.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def pack_age_days() -> Optional[int]:
    """Days since the pack was last built/refreshed (meta.updated_at, else mtime)."""
    ts = pack_meta().get("updated_at")
    if ts:
        try:
            return (datetime.date.today() - datetime.date.fromisoformat(ts[:10])).days
        except ValueError:
            pass
    f = config.ROUTING_DIR / "routing_pack.jsonl"
    if f.exists():
        return (datetime.date.today()
                - datetime.date.fromtimestamp(f.stat().st_mtime)).days
    return None


def find_new(committees: Optional[List[str]] = None) -> List[Tuple[str, str, str]]:
    """[(slug, record, label)] present on the (robots-OK) landing pages but not
    yet in the local routing pack — excluding records already known to parse empty
    (meta.skipped), so refresh stays idempotent and doesn't re-fetch them."""
    _, pack = _load_pack()
    have = {(p["committee"], p["record"]) for p in pack}
    skip = set(pack_meta().get("skipped", []))
    out: List[Tuple[str, str, str]] = []
    for slug in (committees or list(config.COMMITTEES)):
        _guard(slug)
        for rec, label in list_records(slug):
            if (slug, rec) not in have and f"{slug}/{rec}" not in skip:
                out.append((slug, rec, label))
    return out


def refresh(committees: Optional[List[str]] = None, dry_run: bool = False,
            model=None) -> dict:
    new = find_new(committees)
    if dry_run or not new:
        return {"new": len(new), "records": [(s, r) for s, r, _ in new],
                "applied": False}

    model = model or _embedder()
    vecs, pack = _load_pack()
    skipped = set(pack_meta().get("skipped", []))
    add_vecs: List[np.ndarray] = []
    for i, (slug, rec, label) in enumerate(new):
        path = config.RAW_DIR / slug / f"{rec}.html"
        if not (path.exists() and path.stat().st_size > 0):
            if i:
                time.sleep(config.REQUEST_DELAY_SEC)      # polite between fetches
            fetch_record(slug, rec)                       # robots-guarded
        url = f"{config.BASE_URL}/{slug}/{rec}.html"
        m = parse_html(path.read_text(encoding="utf-8"), slug=slug, rec=rec, url=url)
        chs = chunk_meeting(m)
        if not chs:
            skipped.add(f"{slug}/{rec}")   # parses empty (variant template); don't retry
            continue
        emb = model.encode([f"passage: {c.text}" for c in chs],
                           normalize_embeddings=True, show_progress_bar=False,
                           convert_to_numpy=True).astype(np.float32)
        v = emb.mean(axis=0)
        n = float((v * v).sum() ** 0.5)
        add_vecs.append(v / n if n > 1e-9 else v)
        speakers = list(dict.fromkeys(c.speaker_name for c in chs if c.speaker_name))
        pack.append({"committee": slug, "record": rec, "url": url,
                     "date": m.meeting_date or _jp_date(label),
                     "session": _session(rec), "speakers": speakers})

    if add_vecs:
        add = np.vstack(add_vecs).astype(np.float32)
        vecs = add if vecs.size == 0 else np.vstack([vecs, add])

    config.ROUTING_DIR.mkdir(parents=True, exist_ok=True)
    np.save(config.ROUTING_DIR / "routing_vectors.npy", vecs)
    with (config.ROUTING_DIR / "routing_pack.jsonl").open("w", encoding="utf-8") as fh:
        for p in pack:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    meta = pack_meta()
    meta.update(records=len(pack), dim=int(vecs.shape[1]),
                embedding_model=config.E5_MODEL_ID,
                updated_at=datetime.date.today().isoformat(),
                skipped=sorted(skipped),
                note="facts + MOBIUS-derived record vectors only; no minutes text")
    (config.ROUTING_DIR / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"new": len(new), "added": len(add_vecs), "skipped": len(skipped),
            "total": len(pack), "applied": True}
