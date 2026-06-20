"""Robots-respecting, local-only fetcher for 都議会 committee 速記録.

Refuses any slug outside the robots-permitted allowlist (config.ALLOWED_SLUGS),
so the legacy romaji paths and the DB-Search tree can never be auto-fetched.
"""
from __future__ import annotations

import re
import time
import urllib.request
from pathlib import Path
from typing import List, Tuple

from . import config


def _get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": config.USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", errors="replace")


def _guard(slug: str) -> None:
    if slug not in config.ALLOWED_SLUGS:
        raise PermissionError(
            f"slug {slug!r} is not in the robots-permitted allowlist "
            f"{sorted(config.ALLOWED_SLUGS)}. Legacy romaji slugs and the "
            f"DB-Search tree are robots.txt Disallow and will not be fetched.")


def list_records(slug: str) -> List[Tuple[str, str]]:
    """Return [(record_id, date_label)] from a committee landing page."""
    _guard(slug)
    html = _get(f"{config.BASE_URL}/{slug}/")
    out: List[Tuple[str, str]] = []
    for m in re.finditer(r'href="(?:\./)?(\d{4}-\d+)\.html"[^>]*>(.*?)</a>',
                         html, flags=re.S):
        label = re.sub(r"\s+", "", re.sub(r"<[^>]+>", "", m.group(2)))
        out.append((m.group(1), label))
    return out


def fetch_record(slug: str, rec: str) -> Path:
    """Fetch one record to RAW_DIR/<slug>/<rec>.html and return the path."""
    _guard(slug)
    html = _get(f"{config.BASE_URL}/{slug}/{rec}.html")
    dest = config.RAW_DIR / slug / f"{rec}.html"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(html, encoding="utf-8")
    return dest


def fetch_many(slug: str, recs: List[str]) -> List[Path]:
    paths: List[Path] = []
    for i, rec in enumerate(recs):
        if i:
            time.sleep(config.REQUEST_DELAY_SEC)
        paths.append(fetch_record(slug, rec))
        print(f"  fetched {slug}/{rec}.html")
    return paths
