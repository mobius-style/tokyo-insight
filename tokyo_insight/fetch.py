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
    """Return [(record_id, label)] for a committee.

    Standard committees: /record/<slug>/ lists <year>-<seq>.html.
    Year-dir committees (e.g. 予算特別委員会): /record/<slug>/<year>/ lists
    <n>-<mm>.html speaker-segments; record_id is "<year>/<n>-<mm>"."""
    _guard(slug)
    if slug in config.YEAR_DIR_SLUGS:
        return _list_year_dir(slug)
    html = _get(f"{config.BASE_URL}/{slug}/")
    out: List[Tuple[str, str]] = []
    for m in re.finditer(r'href="(?:\./)?(\d{4}-\d+)\.html"[^>]*>(.*?)</a>',
                         html, flags=re.S):
        label = re.sub(r"\s+", "", re.sub(r"<[^>]+>", "", m.group(2)))
        out.append((m.group(1), label))
    return out


def _list_year_dir(slug: str) -> List[Tuple[str, str]]:
    landing = _get(f"{config.BASE_URL}/{slug}/")
    years = sorted(set(re.findall(r'href="(\d{4})/"', landing)))
    out: List[Tuple[str, str]] = []
    for i, y in enumerate(years):
        if i:
            time.sleep(config.REQUEST_DELAY_SEC)
        yh = _get(f"{config.BASE_URL}/{slug}/{y}/")
        for m in re.finditer(r'href="(\d+-\d+)\.html"[^>]*>(.*?)</a>', yh, flags=re.S):
            label = re.sub(r"\s+", "", re.sub(r"<[^>]+>", "", m.group(2)))
            out.append((f"{y}/{m.group(1)}", label or y))
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
